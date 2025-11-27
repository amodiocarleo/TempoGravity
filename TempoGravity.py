import numpy as np
import math 
import subprocess as sub
import os
import emcee
import corner
import matplotlib.pyplot as plt
import multiprocessing as mp
import tempfile
import ast  
import warnings
import sympy as sp  
import random
import re
import argparse 



# Store up to 5 generated .par files in the current directory
saved_par_files = []
max_saved_files = 5  

# General Constants 
light_c = 299792458.                                  # in m/s
M_solar = 1.9884099021470415e30                       # in Kg
G_Newton = 6.67430e-11                                # in m^3/kg/s
cFactor = ((24*3600*light_c**3)/(M_solar*G_Newton))   # it converts Pb/m into a dimensionless quantity when Pb is expressed in days and m in solar masses


# ---------------------------- USER DEFINED ---------------------------------------------------------------------------
# 1) User-set global constants (not sampled, not derived)
# Put values here for parameters you want fixed (examples shown). NB. Case-sensitive !! 
CONSTANTS = {
    "s2": 0.0,
    "s2primo": 0.0,
    "cFactor": cFactor,
    "omega":  120.458*(math.pi/180),
   
    # add more constants here if needed, e.g. "SINI": 0.9
}

# 2) Aliases map: map short names used in equations to the actual parameter names
# Example: map 'e' used in equations to the sampled parameter 'ECC' . NB. Case-sensitive  !! 
ALIASES = {
    "e": "ECC",        # maps symbol 'e' in equations -> sampled parameter 'ECC'
    # "ecc": "ECC",    # alternative alias you might prefer
}

# ---------------------------------------------------------------------------------------------------------------------








def hms_to_seconds(hms):
    """Converts hh:mm:ss.sssss to total seconds."""
    h, m, s = map(float, hms.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_hms(seconds):
    """Converts total seconds back to hh:mm:ss.sssss format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:.7f}"

def dms_to_arcseconds(dms):
    """Converts dd:mm:ss.sssss to total arcseconds."""
    d, m, s = map(float, re.split('[: ]', dms))
    sign = -1 if d < 0 else 1
    return sign * (abs(d) * 3600 + m * 60 + s)

def arcseconds_to_dms(arcseconds):
    """Converts total arcseconds back to dd:mm:ss.sssss format."""
    sign = '-' if arcseconds < 0 else ''
    arcseconds = abs(arcseconds)
    d = int(arcseconds // 3600)
    m = int((arcseconds % 3600) // 60)
    s = arcseconds % 60
    return f"{sign}{d:02d}:{m:02d}:{s:.5f}"






#--------------------------------------------------------------------------------
    
    
    
    
def read_theory_priors(theory_priors_file):
    """Reads additional uniform priors and PK equations from an external file, ignoring comment lines."""
    theory_priors = {}
    equations = {}

    with open(theory_priors_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty and comment lines
                continue

            if "=" in line:  # Equation definition
                try:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        equation = parts[1].strip()
                        equations[param_name] = equation
                    else:
                        print(f"⚠️ Skipping malformed equation line: {line}")
                except ValueError:
                    print(f"⚠️ Skipping malformed equation line: {line}")
            else:  # Assume it's a prior definition
                try:
                    parts = line.split()
                    if len(parts) == 3:
                        param_name = parts[0]
                        lower_bound, upper_bound = float(parts[1]), float(parts[2])
                        theory_priors[param_name] = (lower_bound, upper_bound)
                    else:
                        print(f"⚠️ Skipping malformed prior line: {line}")
                except ValueError:
                    print(f"⚠️ Skipping malformed prior line: {line}")

    return theory_priors, equations    


#--------------------------------------------------------------------------------



def get_priors(par_file, delta, theory_priors_file=None):  
    """Extracts priors from the .par file, includes theory priors if provided.
    
    Modification: 
    - Now, priors from `theory_priors.txt` take precedence over `.par` file priors.
    """

    priors = []
    param_names = []
    sampled_params = {}
    theoretical_params = {}
    processed_jumps = set()  #  Track JUMPs to avoid duplicates

    # Read theory priors FIRST to give them precedence
    additional_priors, equations = {}, {}
    if theory_priors_file and os.path.isfile(theory_priors_file):  # Check if the theory file exists: ---------------
        additional_priors, equations = read_theory_priors(theory_priors_file)
    else:
        warnings.warn("Theory priors file not provided. Only using priors from the .par file.", category=RuntimeWarning)

    with open(par_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            param_name = parts[0]
            
            # Skip parameters defined as equations (they are derived)
            if param_name in equations:
                continue
            
            #  Skip parameters that already have priors from theory_priors.txt
            if param_name in additional_priors:
                continue  

            value = parts[1]

            # Handle RAJ and DECJ 
            if param_name == "RAJ" and len(parts) >= 4 and parts[2] == "1":
                value_sec = hms_to_seconds(value)
                uncertainty = float(parts[3])
                priors.append(("uniform", value_sec - delta * uncertainty, value_sec + delta * uncertainty))  # Uniform instead of Gaussian
                param_names.append("RAJ")
                sampled_params["RAJ"] = value_sec
                continue
            
            if param_name == "DECJ" and len(parts) >= 4 and parts[2] == "1":
                value_arcsec = dms_to_arcseconds(value)
                uncertainty = float(parts[3])
                priors.append(("uniform", value_arcsec - delta * uncertainty, value_arcsec + delta * uncertainty))  # Uniform instead of Gaussian
                param_names.append("DECJ")
                sampled_params["DECJ"] = value_arcsec
                continue

            # Handle JUMP parameters
            if param_name == "JUMP" and len(parts) >= 5 and parts[4] == "1":
                jump_id = f"JUMP_{parts[1]}_{parts[2]}"
                if jump_id in processed_jumps:
                    continue  
                processed_jumps.add(jump_id)
                value, uncertainty = float(parts[3]), 0.1 * abs(float(parts[3]))
                priors.append(("uniform", value - delta * uncertainty, value + delta * uncertainty))  #  Uniform instead of Gaussian
                param_names.append(jump_id)
                sampled_params[jump_id] = value
                continue
            
            # Handle T2EFAC parameters (Uniform prior between 0.1 and 2)
            if param_name == "T2EFAC" and len(parts) >= 4:
                backend_name = parts[2]
                t2efac_key = f"T2EFAC_{backend_name}"     # e.g., "T2EFAC_mkt_S"
                sampled_value = random.uniform(0.1, 2.0)
                sampled_params[t2efac_key] = sampled_value
                priors.append(("uniform", 0.1, 2.0))
                param_names.append(t2efac_key)
                continue
               
               
            # Handle general parameters with fitting flag "1"
            if len(parts) >= 4 and parts[2] == "1":
                value, uncertainty = float(parts[1]), float(parts[3])
                priors.append(("uniform", value - delta * uncertainty, value + delta * uncertainty))  #  Uniform instead of Gaussian
                param_names.append(param_name)
                sampled_params[param_name] = value

    # Apply Theory Priors (which now take precedence)
    for param, (lower, upper) in additional_priors.items():               
        if param in param_names:
            # Overwrite the existing prior from .par with the theory prior
            idx = param_names.index(param)
            priors[idx] = ("uniform", lower, upper)  # Replace prior
            sampled_params[param] = (lower + upper) / 2  # Midpoint as initial value
        else:
            # Add a new prior if the parameter was not in .par
            priors.append(("uniform", lower, upper))
            param_names.append(param)
            sampled_params[param] = (lower + upper) / 2

    # Store equations for theoretical parameters
    theoretical_params = equations.copy()
        

    return sampled_params, priors, param_names, theoretical_params







#--------------------------------------------------------------------------------



def compute_derived_parameters(sampled_params, theoretical_params):
    """Computes derived parameters using sympy, skipping invalid samples (e.g., division by zero, sqrt of negative)."""
    import math
    derived_values = {}
    local_vars = {k: sp.Symbol(k) for k in sampled_params.keys()}
    
   
    # Symbolic math: In equations in the priors file, use pi, sqrt(),sin() etc.. and not math.sqrt() etc..    
    local_vars["pi"]   = sp.pi
    local_vars["sqrt"] = sp.sqrt
    local_vars["sin"]  = sp.sin
    local_vars["cos"]  = sp.cos
    local_vars["log"]  = sp.log
    local_vars["exp"]  = sp.exp
    
    
    
    
    
    
    
    
    for ck in CONSTANTS.keys():
        local_vars[ck] = sp.Symbol(ck)
    


    subs_dict = {k: v for k, v in sampled_params.items()}
    # Add physical constants to the substitution dictionary, otherwise they are considered symbols by sympy
    subs_dict["G_Newton"] = G_Newton
    subs_dict["Msun"] = M_solar
    subs_dict["light_c"] = light_c

    
 #   print("[DEBUG] Is s2 in sampled_params?", "s2" in sampled_params)
 #   print("[DEBUG] Is s2 in subs_dict before alias/constants?", subs_dict.get("s2", None))
    
    # Add user-defined global constants so theoretical expressions can use them  (correction_2)
    for ck, cv in CONSTANTS.items():  
        subs_dict[ck] = cv
        
    
        
        
    # Add alias mappings so short names in equations map to real sampled parameters
    # e.g. if ALIASES = {"e": "ECC"} then subs_dict["e"] = subs_dict["ECC"] (if ECC exists)
    for alias_name, real_name in ALIASES.items():
        if real_name in subs_dict:
            subs_dict[alias_name] = subs_dict[real_name]
        else:
            # If real_name not present, we do not set alias and we print a warning
            # (keeps behavior explicit and safe)
            print(f"⚠️ ALIAS WARNING: '{real_name}' not found among sampled parameters; alias '{alias_name}' not set.")    


    """
    # ---------------- DEBUG START ----------------
    if "ECC" in subs_dict and "e" in subs_dict:
        print("\n[DEBUG] ECC sampled value =", subs_dict["ECC"])
        print("[DEBUG] e alias value     =", subs_dict["e"])
        print("[DEBUG] (They should be identical!)\n")
    else:
        print("\n[DEBUG] WARNING: 'ECC' or 'e' missing in subs_dict\n")
    # ---------------- DEBUG END ----------------
    """
    
    
    
    
    
    


    for param, equation in theoretical_params.items():
        try:
        
            # ------------------- DEGUB FOR GAMMA ------------------------------
            """
            if param == "GAMMA":
                print("\n--- DEBUG: Evaluating GAMMA for this sample ---")
                for key in ["ECC", "PB", "m1", "m2"]:
                    if key in sampled_params:
                        print(f"{key} = {sampled_params[key]}")
                    else:
                        print(f"{key} = NOT FOUND in sampled_params")

                print(f"Equation used: {equation}")
            """    
           # --------------------- END DEBUG FOR GAMMA -------_-----------------    
        
        
        
        
        
        
        
        
        
        
            expr = sp.sympify(equation, locals=local_vars)
            evaluated_value = expr.subs(subs_dict)
            
            # --------------------- DEBUG: print raw evaluated parameter result ----------------------------
            
            """
            if param == "GAMMA":
               print(f"DEBUG : Raw evaluated GAMMA = {evaluated_value}")
            """
            # ---------------------- END DEBUG  ---------------------------
            

            # Check for invalid values
            if evaluated_value.is_real:
                evaluated_value = float(evaluated_value)
                if not np.isfinite(evaluated_value):  # Checks for NaN, inf, or -inf
                    raise ValueError(f"Invalid value ({evaluated_value}) in {param}")
                if isinstance(evaluated_value, complex):  # Avoid complex numbers
                    raise ValueError(f"Complex number encountered in {param}")
            else:
                raise ValueError(f"Non-real result in {param}")

            derived_values[param] = evaluated_value

        except (ZeroDivisionError, ValueError, sp.SympifyError) as e:
            print(f"⚠️ Skipping sample due to invalid computation in {param}: {e}")
            return None  # Signal to skip this sample

    return derived_values




#-------------------------------------------------------------------------------------------------------







def sample_from_priors(priors):
    sampled_values = []
    
    for i, prior in enumerate(priors):
        if prior[0] == "gaussian":
            value = np.random.normal(prior[1], prior[2])
        elif prior[0] == "uniform":
            value = np.random.uniform(prior[1], prior[2])
            # Enforce T2EFAC range
            if "T2EFAC" in param_names[i]:  # Assuming param_names is accessible
                value = np.clip(value, 0.1, 2.0)
        sampled_values.append(value)
        
        #Debug: Print sampled values to check for anomalies
    #    print(f"DEBUG: Sampled value for {prior}: {value}")

    return sampled_values





#-------------------------------------------------------------------------------------------------------



def write_new_par_file(par_file, new_par_file, sampled_params, theoretical_params):
    """
    Writes a new .par file with updated sampled and derived parameter values,
    preserving structure, comments, and formatting from the original .par file.
    
    - Updates only the **values** of `T2EFAC` and `JUMP` in the correct column (4th column)
    - Keeps all other lines unchanged (including comments and spacing)
    - Skips iteration if T2EFAC is negative
    """
    global saved_par_files
    
    # Read the original .par file
    with open(par_file, "r") as f:
        lines = f.readlines()
    
       
    # Use provided derived parameter values if present (avoid recomputing).
    # sampled_params may already contain derived parameter keys (because we pass full_params from log_prob).
    derived_values = {k: sampled_params[k] for k in theoretical_params.keys() if k in sampled_params}

    # If theoretical_params exist but no derived values were provided in sampled_params,
    # fall back to computing them (keeps function robust for standalone use).
    if theoretical_params and not derived_values:
        derived_values = compute_derived_parameters(sampled_params, theoretical_params)
        if derived_values is None:
            print("⚠️ Skipping iteration due to invalid derived parameters.")
            return  # Skip iteration if derived parameters are invalid
        
    new_lines = []
    
    for line in lines:
        original_line = line.strip()  # Preserve original formatting
        parts = line.strip().split()
        if len(parts) == 0:
            new_lines.append(original_line)  # Keep empty lines
            continue

        param_name = parts[0]
        
        # Handle T2EFAC parameters
        if param_name == "T2EFAC" and len(parts) >= 4:
            backend_name = parts[2]  # e.g., "mkt_L"
            t2efac_key = f"T2EFAC_{backend_name}"  # Ensure correct key usage
            
            if t2efac_key in sampled_params:
                new_value = sampled_params[t2efac_key]
           #     print(f"✅ DEBUG: Before writing to .par → T2EFAC[{backend_name}] = {new_value}")
                
                # Skip iteration if T2EFAC is negative
                if new_value < 0:
                    print(f"⚠️ ERROR: Negative T2EFAC detected ({new_value}) → Skipping iteration.")
                    return
                
                parts[3] = f"{new_value:.21g}"  # Update only the value
                new_line = " ".join(parts)  # Keep original structure
            else:
                new_line = original_line  # Keep unchanged
        
        # Handle JUMP parameters
        elif param_name == "JUMP" and len(parts) >= 4:
            jump_key = f"JUMP_{parts[1]}_{parts[2]}"  # Ensure correct key usage
            if jump_key in sampled_params:
                parts[3] = f"{sampled_params[jump_key]:.21g}"  # Update value
                new_line = " ".join(parts)  # Keep format
            else:
                new_line = original_line  # Keep unchanged
                
        # Handle RAJ and DECJ         
        elif param_name == "RAJ" and "RAJ" in sampled_params:
            new_value = seconds_to_hms(sampled_params["RAJ"])
            parts[1] = new_value
            new_line = " ".join(parts)  # Keep structure
        elif param_name == "DECJ" and "DECJ" in sampled_params:
            new_value = arcseconds_to_dms(sampled_params["DECJ"])
            parts[1] = new_value        
            new_line = " ".join(parts)  # Keep structure
            
        # Handle other parameters normally
        elif param_name in sampled_params:
            parts[1] = f"{sampled_params[param_name]:.21g}"  # Update value
            new_line = " ".join(parts)  # Keep structure
        
        # Handle derived parameters
        elif param_name in derived_values:
            parts[1] = f"{derived_values[param_name]:.21g}"  # Update value
            new_line = " ".join(parts)  # Keep structure
        
        else:
            new_line = original_line  # Keep unchanged lines
        
        new_lines.append(new_line)
    
    # Write the updated .par file
    with open(new_par_file, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    
    # Save the first 5 generated .par files for debugging
    if len(saved_par_files) < max_saved_files:
        save_path = os.path.join(os.getcwd(), f"sample_{len(saved_par_files)+1}.par")
        with open(save_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        saved_par_files.append(save_path)
    
  #  print(f"✅ DEBUG: Successfully wrote new .par file: {new_par_file}")
   
   
   # Debug print: Show contents of the new .par file                                             # .......  PRINT CURRENT PAR FILES  (debug)........
   
 #   print(f"✅ DEBUG: Contents of '{new_par_file}':")
 #   for line in new_lines:
 #       print(line)


        
        
#----------------------------------------------------------------------------------------------------------------        



    
    

    
    
    

def run_tempo2(par_file, tim_file, output_file="residuals.dat"):
    """
    Runs TEMPO2 with the given .par and .tim files.
    If TEMPO2 fails, the .par file is saved in a 'failed_pars' directory for debugging.
    """
    
    cmd = f"tempo2 -nofit -output general2 -outfile {output_file} -s '{{sat}} {{post}}\n' -f {par_file} {tim_file}"
    
    
    
    result = sub.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"⚠️ TEMPO2 Error:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")  # Print full error output
        
        # Ensure the 'failed_pars' directory exists
        failed_dir = "failed_pars"
        if not os.path.exists(failed_dir):
            os.makedirs(failed_dir)

        # Create a unique filename for the failed .par file
        failed_par_file = os.path.join(failed_dir, f"failed_{os.path.basename(par_file)}")
        
        # Save the failed .par file
        try:
            with open(par_file, "r") as original, open(failed_par_file, "w") as copy:
                copy.write(original.read())
            print(f"🚨 TEMPO2 failed! The corresponding .par file has been saved to: {failed_par_file}")
        except Exception as e:
            print(f"⚠️ ERROR: Failed to save the problematic .par file: {e}")

        return None  # Return None to indicate failure

    return output_file
    
    

    
#----------------------------------------------------------------------------------------------------------------
    
def compute_likelihood(residuals, uncertainties):
    if np.any(uncertainties <= 0):
        print(f"⚠️ Warning: Found non-positive uncertainties! Min value = {uncertainties}")
    
    chi2 = np.sum((residuals / uncertainties) ** 2)
 #   print(f"\n DEBUG: The CHI2 is this: {chi2:.2f}")
    
    # Replace invalid uncertainties with a small value to avoid log issues
    safe_uncertainties = np.maximum(uncertainties, 1e-20)

    
    log_likelihood = -0.5 * chi2 - np.sum(np.log(safe_uncertainties))
    return log_likelihood if np.isfinite(log_likelihood) else -np.inf
 
#-------------------------------------------------------------------------------------------------------------    

    
#--------------------------------------------------------------------------------------------------------------   


warned_backends = set()  # Initialize OUTSIDE load_residuals ! 

def load_residuals(residual_file, tim_file, new_par_file): 
    """
    Loads residuals and uncertainties, applying T2EFAC corrections.

    Fixes:
    - Handles empty residual files gracefully.
    - Uses a dictionary for fast O(1) MJD lookups instead of O(N²) looping.
    - Fixes incorrect `T2EFAC` extraction from the parameter file.
    - Handles missing TOA matches correctly with warnings (now only once per backend).
    """

    global warned_backends 

    try:
        if os.stat(residual_file).st_size == 0:
            raise ValueError("Residual file is empty.")
        residuals_data = np.loadtxt(residual_file, usecols=(0, 1))
    except FileNotFoundError:
        print(f"Error: Residual file '{residual_file}' not found.")
        return None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None
    except IndexError:
        print("Error: Residual file format is incorrect.")
        return None, None

    mjd_uncertainties = []     # Stores (MJD, uncertainty_IN_SECONDS, telescope_flag) # Modified comment
    try:
        with open(tim_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or line.startswith("C"):
                    continue

                if len(parts) >= 7:  # Assumed format: Name,Freq,TOA,uncertainty,telescope,-i,backend
                    mjd = round(float(parts[2]), 11)  # Round MJD (column 3) to 12 decimal
                    
                    # --- MODIFIED PART: Convert uncertainty from microseconds to seconds ---
                    uncertainty_us = float(parts[3]) # Uncertainty from TIM file (in microseconds)
                    uncertainty_s = uncertainty_us / 1_000_000.0 # Convert to seconds
                    # --- END OF MODIFIED PART ---
                    
                    backend = parts[6]               # Extract backend (column 7)
                    if uncertainty_s <= 0: # Check using the seconds value
                        print(f"⚠️ Invalid uncertainty for MJD={mjd}: {uncertainty_us} us (or {uncertainty_s} s)")
                    mjd_uncertainties.append((mjd, uncertainty_s, backend)) # Store uncertainty in seconds
    except FileNotFoundError:
        print(f"Error: TIM file '{tim_file}' not found.")
        return None, None
    except (ValueError, IndexError): # Added specific exception types for clarity
        print(f"Error: Invalid format in TIM file or issue processing a line: '{line.strip()}'")
        return None, None
    
    # Read the T2EFAC values from the updated .par file
    t2efac_values = {}
    with open(new_par_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "T2EFAC":
                backend_name = parts[2]          # Extract backend name (e.g., "mkt_L")
                efac_value = float(parts[3])   # Store using backend name
                t2efac_values[backend_name] = efac_value
            #    print(f"✅ DEBUG: Read T2EFAC[{backend_name}] = {efac_value} from {new_par_file}")


    mjd_unc_dict = {mjd: (unc_s, backend) for mjd, unc_s, backend in mjd_uncertainties} # unc_s is now in seconds
    matched_residuals, matched_uncertainties = [], []
  #  warned_backends = set()  # Keep track of backends warned about # This line is redundant as global warned_backends is used

    for mjd_residual, res in residuals_data:
        mjd_residual = round(mjd_residual, 11)
        if mjd_residual in mjd_unc_dict:
            unc_seconds, backend = mjd_unc_dict[mjd_residual] # unc_seconds is already in seconds
            efac_value = t2efac_values.get(backend, None)
            if efac_value is None and backend not in warned_backends:  # Check if we've warned
                print(f"⚠️ Warning: No T2EFAC OOOO found for backend '{backend}'. Assuming EFAC = 1.0")
                warned_backends.add(backend)  # Add to the set
                efac_value = 1.0        # Default correction
            elif efac_value is None:
                efac_value = 1.0  # Still use the default, but don't warn again

            matched_residuals.append(res)
            # Apply EFAC to uncertainty already in seconds
            matched_uncertainties.append(unc_seconds * efac_value) 
          #  print(f"DEBUG: Matched Residuals:{matched_residuals} \nMatched Uncertainities:{matched_uncertainties}")
        else:
            print(f"⚠️ Warning: No TOA match found for residual MJD={mjd_residual}. Skipping.")

    return np.array(matched_residuals), np.array(matched_uncertainties)








#------------------------------------------------------------------------------------------------------------ 

"""
    
    
def log_prob(theta, par_file, tim_file, param_names, theoretical_params, priors):       ######### added priors
    sampled_params = {name: value for name, value in zip(param_names, theta)}
    
    
    # Enforce uniform prior bounds manually                                            #########  added block
    for i, name in enumerate(param_names):
        prior_type, lower, upper = priors[i]
        if prior_type == "uniform":
            if not (lower <= sampled_params[name] <= upper):
                return -np.inf  # Reject samples outside prior bounds

    
    
    t2efac_values = {param: value for param, value in sampled_params.items() if "T2EFAC" in param}   # DEBUG LINE 1
  #  print(f"✅ DEBUG: MCMC sampled T2EFAC values: {t2efac_values}")                                 # DEBUG LINE 2


    
    # Check for invalid T2EFAC values
    for param, value in sampled_params.items():
        if "T2EFAC" in param and (value < 0.1 or value > 2.0):
            return -np.inf  # Reject the sample if T2EFAC is outside the valid range
    
    # Compute derived parameters
    derived_params = compute_derived_parameters(sampled_params, theoretical_params)
    if derived_params is None:
        return -np.inf  # Reject the sample
    
    # Merge both sampled and derived parameters
    full_params = {**sampled_params, **derived_params}

    # Write new par file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_par:
        new_par_file = tmp_par.name

    write_new_par_file(par_file, new_par_file, full_params, theoretical_params)

    # Run TEMPO2
    
    #print(f"DEBUG: Running TEMPO2 with {new_par_file}")    # Print which .par file is currently used
    #print(f"DEBUG: Sampled parameters: {sampled_params}")  # Print current parameter values
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    result = run_tempo2(new_par_file, tim_file, output_file)

    if result is None:
        print("⚠️ Warning: TEMPO2 may not have run successfully!")
        print(f"DEBUG: Sampled parameters that caused failure: {sampled_params}")
        return -np.inf
     
     
    # Load Residuals 
    residuals, uncertainties = load_residuals(output_file, tim_file, new_par_file)
    if residuals is None or uncertainties is None:
        return -np.inf

     # Compute the logliklihood
    log_likelihood = compute_likelihood(residuals, uncertainties)

    try:
        os.remove(new_par_file)
        if os.path.exists(output_file):
            os.remove(output_file)
        else:
            print("Warning: residual file not found before deletion")         
    except OSError:
        pass

    return log_likelihood    
"""    
#-----------------------------------------------------------------------------------------------------------




def log_prob(theta, par_file, tim_file, param_names, theoretical_params, priors):
    """
    Compute log-probability for a parameter vector theta.
    Returns (log_likelihood, derived_blob) so emcee stores derived_blob as the per-sample blob.
    On failure or prior rejection returns (-np.inf, nan_blob) where nan_blob has same length as derived_keys.
    """
    sampled_params = {name: value for name, value in zip(param_names, theta)}

    # Prepare stable derived_keys order (same used in post-processing)
    derived_keys = list(theoretical_params.keys())
    nan_blob = np.full(len(derived_keys), np.nan, dtype=float)

    # Enforce uniform prior bounds manually
    for i, name in enumerate(param_names):
        prior_type, lower, upper = priors[i]
        if prior_type == "uniform":
            if not (lower <= sampled_params[name] <= upper):
                # Return numeric nan blob (do NOT return None)
                return -np.inf, nan_blob

    # Basic T2EFAC sanity (if present in names)
    for param, value in sampled_params.items():
        if "T2EFAC" in param and (value < 0.1 or value > 10.0):
            return -np.inf, nan_blob

    # Compute derived parameters using your existing SymPy-based function
    derived_params = compute_derived_parameters(sampled_params, theoretical_params)
    if derived_params is None:
        # keep alignment by returning nan_blob
        return -np.inf, nan_blob

    # Merge both sampled and derived parameters for writing .par
    full_params = {**sampled_params, **derived_params}

    # Write new par file (use your existing helper)
    tmp_par_file_obj = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".par")
    new_par_file = tmp_par_file_obj.name
    tmp_par_file_obj.close()
    write_new_par_file(par_file, new_par_file, full_params, theoretical_params)

    # Prepare tempo2 output file
    tmp_output_file_obj = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".dat")
    output_file = tmp_output_file_obj.name
    tmp_output_file_obj.close()

    # Run TEMPO2
    tempo2_result = run_tempo2(new_par_file, tim_file, output_file)

    # If tempo2 failed, clean up and return nan_blob
    if tempo2_result is None:
        print(f"WARNING: Tempo2 returned None")
        try:
            if os.path.exists(new_par_file): os.remove(new_par_file)
            if os.path.exists(output_file): os.remove(output_file)
        except Exception:
            pass
        return -np.inf, nan_blob

    # Load residuals and uncertainties
    residuals, uncertainties = load_residuals(output_file, tim_file, new_par_file)
    if residuals is None or uncertainties is None or len(residuals) == 0:
        try:
            if os.path.exists(new_par_file): os.remove(new_par_file)
            if os.path.exists(output_file): os.remove(output_file)
        except Exception:
            pass
        return -np.inf, nan_blob

    # Compute likelihood using your existing function
    log_likelihood_val = compute_likelihood(residuals, uncertainties)

    # Cleanup temporary files
    try:
        if os.path.exists(new_par_file): os.remove(new_par_file)
        if os.path.exists(output_file): os.remove(output_file)
    except OSError:
        pass

    # Build numpy blob in the same derived_keys order
    # If some keys missing in derived_params, put NaN there (defensive)
    derived_blob = np.array([float(derived_params[k]) if (k in derived_params and
                              np.isfinite(derived_params[k])) else np.nan
                              for k in derived_keys], dtype=float)

    # Return (log_prob, blob) as required by emcee
    return float(log_likelihood_val), derived_blob




  
    
    
#------------------------------------------------------------------------------------------------------------    
"""
def run_emcee(par_file, tim_file, n_samples, n_walkers, n_threads, delta, theoretical_params, theory_priors_file):    # Removed None from theory_priors_file ---------
 
   

    # Extract priors and parameter names
    sampled_params, priors, param_names, theoretical_params = get_priors(par_file, delta, theory_priors_file)

    ndim = len(param_names)

    # Initialize walkers with correctly mapped parameter values
    initial_pos = np.array([sample_from_priors(priors) for _ in range(n_walkers)])
    
    
    # --- ADD THIS BLOCK TO SAVE INITIAL POSITIONS TO A FILE ---
    output_filename = "initial_walker_positions.txt"
    print(f"\n--- Saving Initial Walker Positions to '{output_filename}' ---")

    with open(output_filename, 'w') as f:
        # Write the header with parameter names
        header = "# " + "\t".join(param_names) # Use tab for separation, add '#' for comment
        f.write(header + "\n")

        # Write each walker's position
        # Using np.savetxt for cleaner output and better control
        np.savetxt(f, initial_pos, fmt='%.8e', delimiter='\t') # '%.8e' for scientific notation with 8 decimals, tab-delimited
                                                              # fmt='%.12f' for fixed-point notation with 12 decimals (adjust as needed)
    
    

   

    with mp.Pool(n_threads) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_prob, 
            args=(par_file, tim_file, param_names, theoretical_params, priors),      ######### added priors 
            pool=pool
        )
        print("Running MCMC...")
        sampler.run_mcmc(initial_pos, n_samples, progress=True)

    # Extract MCMC samples
 #   samples = sampler.get_chain(flat=True)
    burn_in = int(0.1 * n_samples) # Example: 10% of samples for burn-in 
    samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)                      # discard(num)    ____ BURN-IN _____  


  
    return samples, param_names, sampler
"""

#------------------------------------------------------------------------------------------------------------

def run_emcee(par_file, tim_file, n_samples, n_walkers, n_threads, delta, theoretical_params, theory_priors_file):
    """
    Runs the emcee sampler and prints the autocorrelation time.
    """

    # Extract priors and parameter names
    sampled_params, priors, param_names, theoretical_params = get_priors(par_file, delta, theory_priors_file)

    ndim = len(param_names)

    # Initialize walkers with correctly mapped parameter values
    initial_pos = np.array([sample_from_priors(priors) for _ in range(n_walkers)])

    # Optional: Save initial walker positions (uncomment if needed)
    """
    output_filename = "initial_walker_positions.txt"
    print(f"\n--- Saving Initial Walker Positions to '{output_filename}' ---")
    with open(output_filename, 'w') as f:
        header = "# " + "\t".join(param_names)
        f.write(header + "\n")
        np.savetxt(f, initial_pos, fmt='%.8e', delimiter='\t')
    """

    with mp.Pool(n_threads) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_prob,
            args=(par_file, tim_file, param_names, theoretical_params, priors),
            pool=pool
        )
        print("Running MCMC...")
        sampler.run_mcmc(initial_pos, n_samples, progress=True)

    # --- CALCULATE AND PRINT AUTOCORRELATION TIME ---
    try:
        # It's generally recommended to compute this on the chain *after* potential burn-in,
        # but emcee's get_autocorr_time can be sensitive to chain length.
        # You might want to compute it on the chain before discarding burn-in,
        # or on the chain after discarding burn-in if you have enough samples.
        # The `tol` parameter can be adjusted. Lower `tol` is stricter.
        autocorr_time = sampler.get_autocorr_time(tol=0) # Use tol=0 to get an estimate even if it hasn't converged
        print("\n--- Autocorrelation Time ---")
        for i, name in enumerate(param_names):
            print(f"{name}: {autocorr_time[i]:.2f} steps")

        
    except emcee.autocorr.AutocorrError as e:
        print("\n--- Autocorrelation Time ---")
        print(f"Could not estimate autocorrelation time: {e}")
        print("This might indicate that the chains are too short or have not converged.")
    # --- END AUTOCORRELATION TIME BLOCK ---

    # Extract MCMC samples (after burn-in)
    burn_in = int(0.1 * n_samples)  # Example: 10% of samples for burn-in
#    print(f"\nDiscarding {burn_in} samples for burn-in (out of {n_samples} total samples per walker).")
    samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)

#    print(f"Shape of samples after burn-in and flattening: {samples.shape}")
    if samples.shape[0] < n_walkers: # A basic check
        print("Warning: Number of effective samples after burn-in is less than the number of walkers.")
        print("This could indicate too few total samples or too large a burn-in fraction.")


    return samples, param_names, sampler



#-------------------------------------------------------------------------------------------------------------
"""
def run_emcee(par_file, tim_file, n_samples, n_walkers, n_threads, delta, theoretical_params, theory_priors_file):
    # Removed None from theory_priors_file ---------

   

    # Extract priors and parameter names
    # Note: sampled_params here would be just one sample, which isn't used for initialization anymore
    # but still used to define the parameter space for log_prob args.
    sampled_params_single_instance, priors, param_names, theoretical_params = get_priors(par_file, delta, theory_priors_file)

    ndim = len(param_names)

    # Initialize walkers with correctly mapped parameter values
    # Corrected way to create a 2D numpy array for initial_pos
    initial_pos = np.zeros((n_walkers, ndim)) # Pre-allocate array
    for i in range(n_walkers):
        # Get a dictionary of sampled parameters for each walker
        sampled_dict_for_walker = sample_from_priors(priors)
        # Convert the dictionary to a numpy array, ensuring parameter order
        
        # --- ADD THESE DEBUG PRINTS ---
        print(f"\n--- Debugging Walker {i} ---")
        print(f"Type of sampled_dict_for_walker: {type(sampled_dict_for_walker)}")
        print(f"Content of sampled_dict_for_walker: {sampled_dict_for_walker}")
        print(f"Content of param_names: {param_names}")
        # --- END DEBUG PRINTS ---
        
        initial_pos[i, :] = np.array([sampled_dict_for_walker[p] for p in param_names])

    # Optional: add a tiny perturbation to ensure walkers are not perfectly coincident
    # This can sometimes help emcee start if all points are exactly the same
    initial_pos += np.random.randn(n_walkers, ndim) * 1e-9 # Smaller perturbation

    # Debug print for initial positions
    print("\n--- Initial Walker Positions ---")
    for i, pos in enumerate(initial_pos):
        print(f"Walker {i + 1}:")
        for j, param_value in enumerate(pos):
            print(f"  {param_names[j]}: {param_value}")
    print("--------------------------------\n")


    with mp.Pool(n_threads) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_prob,
            args=(par_file, tim_file, param_names, theoretical_params, priors),
            pool=pool
        )
        print("Running MCMC...")
        # Make sure initial_pos is a 2D numpy array of floats
        sampler.run_mcmc(initial_pos, n_samples, progress=True)

    
    burn_in = int(0.1 * n_samples) # Example: 10% of samples for burn-in
    samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)

    return samples, param_names, sampler
    
    """
#-------------------------------------------------------------------------------------------------------------
       
    
    
    
def plot_posterior(posterior, derived_posterior, param_names):
    """
    Plots the posterior distribution using corner.py, including derived parameters.
    Only specific parameters are plotted if they exist.
    """
    # List of parameters to plot (if they exist)
    global selected_params  # ✅ Access selected_params from the main function
    

    # Combine sampled and derived parameter names
    full_param_names = param_names + list(derived_posterior[0].keys())
    
    # Combine sampled and derived parameter values
    full_posterior = np.hstack((posterior, np.array([list(d.values()) for d in derived_posterior])))

    # Find indices of selected parameters that exist in the data
    indices_to_plot = [full_param_names.index(param) for param in selected_params if param in full_param_names]

    if not indices_to_plot:
        print("⚠️ Warning: None of the selected parameters are present in the dataset. No plot generated.")
        return

    # Extract only the selected parameters for plotting
    filtered_posterior = full_posterior[:, indices_to_plot]
    filtered_param_names = [full_param_names[i] for i in indices_to_plot]  
    
    # Remove all sets containing NaN or infinities to avoid issues in the plotting
  #  filtered_posterior = filtered_posterior[np.isfinite(filtered_posterior).all(axis=1)]

    # Generate corner plot with only the selected parameters
    fig = corner.corner(
        filtered_posterior, labels=filtered_param_names,color="goldenrod",
        plot_density=True, fill_contours=True, bins=20, smooth=1.3,
        levels=[0.68, 0.95, 0.997], show_titles=True, title_fmt=".2E", truth_color="green"
    )
    
    plt.savefig("corner_plot.png", dpi=600)
    plt.close(fig)
    print(f"✅ Corner Plots saved:  corner_plot.png")
   
    
    
#----------------------------------------------------------------------------------------------------------------    

import math

def plot_trace_grid(sampler, param_names, filename="trace_all_params.png"):
    """
    Plots trace plots for all parameters in a single image grid and saves it.
    """
    chain = sampler.get_chain()  # shape: (n_steps, n_walkers, n_dim)
    n_steps, n_walkers, n_dim = chain.shape

    # Use all parameter indices
    num_to_plot = n_dim

    # Grid layout: adjust n_cols to your preference
    n_cols = 3
    n_rows = math.ceil(num_to_plot / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2.5), sharex=True)
    axes = axes.flatten()

    for i in range(num_to_plot):
        ax = axes[i]
        for w in range(n_walkers):
            ax.plot(chain[:, w, i], alpha=0.3, lw=0.5)
        ax.set_title(param_names[i], fontsize=9)
        ax.grid(True)
        if i % n_cols == 0:
            ax.set_ylabel("Value")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Step")

    # Turn off any unused axes
    for j in range(num_to_plot, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"✅ Trace plot saved:  {filename}")



        
#-----------------------------------------------------------------------------------------------------------------    
    





if __name__ == "__main__":
    import sys, time
    
    
    
    # Parse command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python3 script.py file.par file.tim [theory_priors.txt] [n_samples] [n_walkers] [delta]")
        sys.exit(1)
    
    par_file = sys.argv[1]
    tim_file = sys.argv[2]
    
    
    
    # Initialize with default values
    theory_priors_file = None
    n_samples = 10000  # Default value
    n_walkers = 100   # Default value
    delta = 10       # Default value

    # Check for optional arguments in order
    arg_index = 3
    if len(sys.argv) > arg_index:
        # Check if the next argument is a file (e.g., ends with .txt or similar)
        # This is a simple heuristic; a more robust check might be needed for complex cases.
        if sys.argv[arg_index].lower().endswith(('.txt', '.dat', '.yaml')): # Add relevant extensions
            theory_priors_file = sys.argv[arg_index]
            arg_index += 1
        elif sys.argv[arg_index].lower() == 'none': # Allow 'none' as explicit non-file
             theory_priors_file = None
             arg_index += 1
        # If it's not a known file extension, assume it's the start of numerical parameters
        # In this simpler approach, the order matters.

    if len(sys.argv) > arg_index:
        try:
            n_samples = int(sys.argv[arg_index])
            arg_index += 1
        except ValueError:
            print(f"Warning: Could not parse n_samples from '{sys.argv[arg_index]}'. Using default ({n_samples}).")
            arg_index += 1 # Still advance to prevent misinterpretation

    if len(sys.argv) > arg_index:
        try:
            n_walkers = int(sys.argv[arg_index])
            arg_index += 1
        except ValueError:
            print(f"Warning: Could not parse n_walkers from '{sys.argv[arg_index]}'. Using default ({n_walkers}).")
            arg_index += 1

    if len(sys.argv) > arg_index:
        try:
            delta = float(sys.argv[arg_index])
            arg_index += 1
        except ValueError:
            print(f"Warning: Could not parse delta from '{sys.argv[arg_index]}'. Using default ({delta}).")
            # arg_index does not need to advance as it's the last possible argument

    print(f"\nProvided Theory File = {theory_priors_file}")
    print(f"n_samples = {n_samples}")
    print(f"n_walkers = {n_walkers}")
    print(f"delta = {delta}")
    
    
    
    
    
    
  #  theory_priors_file = sys.argv[3] if len(sys.argv) > 3 else None
  #  print(f"Provided Theory File = {theory_priors_file}")

    
    
    
    #############################################################   USER CONFIGURATION   ###########################################
    selected_params = ["PB", "OM", "ECC", "PMRA", "PMDEC", "PBDOT", "H3", "STIG", "OMDOT", "GAMMA", "XDOT","T2EFAC_pks_dfb4","T2EFAC_pks_afb"]  # List of parameters to plot 
#    n_samples = 100    # 10000                                                                                             
#    n_walkers = 52    # at least 4x N_param  
                                         
#    delta = 20         # 5                                                        
    ################################################################################################################################# 
    
     
    ##########################################################   ADDITIONAL CONSTANTS   ##########################################################################
    #omega=  120.458*(math.pi/180)          # 11.33*(math.pi/180)
    
    
    
    
    ##############################################################################################################################################################               

        
    # Extract priors and theoretical parameters
    sampled_params, priors, param_names, theoretical_params = get_priors(par_file, delta, theory_priors_file)
    
   # print("\nSampling the following parameters:")
   # for name, prior in zip(param_names, priors):
   #    if prior[0] == "uniform":
   #        print(f"{name}: Uniform({prior[1]:.8g}, {prior[2]:.8g})")
            
            
            
    print("\nSampling the following parameters:")
    for name, prior in zip(param_names, priors):
        if name not in theoretical_params and prior[0] == "uniform":  # Only print if NOT a derived parameter
            print(f"{name}: Uniform({prior[1]:.10g}, {prior[2]:.10g})")

    
            
    
    if theoretical_params:
        print("\nDerived parameters:")
       # for name in theoretical_params.keys():                # (a) Use this line if you only want to print the name of the derived parameters
        for name, equation in theoretical_params.items():     # (b) Use this line if you want to print both the name and the equation of the derived parameters
        #    print(f"{name}")              # Use this line for option (a)
            print(f"{name} =  {equation}")  # Use this line for option (b)
            
   

    n_threads = mp.cpu_count()
    tot_params=len(param_names)
    
    print(f"\nNumber of CPU cores: {n_threads}")
    print(f"\nTotal number of parameters: {tot_params}")
    
    # Run MCMC sampling
    start_time = time.time()
    #posterior, param_names = run_emcee(par_file, tim_file, n_samples, n_walkers, n_threads, delta, theoretical_params) -----------------------------------------
    posterior, param_names, sampler = run_emcee(par_file, tim_file, n_samples, n_walkers, n_threads, delta, theoretical_params, theory_priors_file)
    
  #  plot_trace_grid(sampler, param_names)                                                                      # ____ TRACE_PLOTS _____ON/OFF

   



    
    #print(f"Posterior shape = (Number of Samples, Number of Params) =  {posterior.shape}")
    
    

    print(f"\n Computing derived posteriors ... ")
    



    

    
        # --------------------- Use stored blobs for derived parameters (fast) ---------------------
    # 'posterior' already is the flattened sample array returned by run_emcee (burn-in removed).
    # We will fetch the blobs stored during sampling (same discard used in run_emcee).
    burn_in = int(0.1 * n_samples)  # same convention used inside run_emcee()
    blobs = sampler.get_blobs(discard=burn_in, flat=True)  # shape: (N_samples_after_burn, n_derived_total)
    blobs_array = np.array(blobs)  # numeric array; rows correspond to rows in get_chain(..., flat=True)

    # Determine derived parameter names and which ones are not already sampled
    derived_keys = list(theoretical_params.keys())     # stable order used in log_prob()
    derived_only = [k for k in derived_keys if k not in param_names]





    
 
        
        



    if blobs_array.size == 0:
    # If there are theoretical (derived) parameters expected, warn and fall back to
    # recomputing them (this is rare and indicates something went wrong storing blobs).
        if theoretical_params:
            print("⚠️ No blobs found in sampler (unexpected). Falling back to recomputing derived parameters.")
            derived_posterior = [compute_derived_parameters(dict(zip(param_names, sample)), theoretical_params) for sample in posterior]
            derived_posterior = [d for d in derived_posterior if d is not None]
            derived_array = np.array([[d[k] for k in derived_only] for d in derived_posterior])
            # Align shapes: posterior may have more rows than derived_array if some were invalid and filtered out.
            n_rows = min(posterior.shape[0], derived_array.shape[0])
            posterior = posterior[:n_rows, :]
            derived_array = derived_array[:n_rows, :]
        else:
            # No theoretical params were provided => no blobs expected; treat derived_array as empty (N x 0)
            derived_array = np.zeros((posterior.shape[0], 0), dtype=float)

    else:
        # Extract columns of interest (derived_only) from blobs (which are in derived_keys order)
        indices = [derived_keys.index(k) for k in derived_only]
        # Align row counts: if any mismatch, trim to the smaller number of rows (shouldn't normally happen)
        n_rows = min(blobs_array.shape[0], posterior.shape[0])
        blobs_array = blobs_array[:n_rows, :]
        posterior = posterior[:n_rows, :]
        derived_array = blobs_array[:, indices]

        
    
    
    
    
    




    # Mask invalid derived rows (where any derived value is NaN). This keeps the chain/sample alignment explicit:
    valid_mask = np.all(np.isfinite(derived_array), axis=1)
    if not np.all(valid_mask):
        num_invalid = np.count_nonzero(~valid_mask)
        print(f"⚠️ Warning: {num_invalid} posterior rows contain invalid derived params (NaN) and will be discarded from the final posterior.")
    posterior_valid = posterior[valid_mask]
    derived_array_valid = derived_array[valid_mask]

    # Build final full posterior = [sampled params | derived_only params]
    full_param_names = param_names + derived_only
    if derived_array_valid.size == 0:
        # No derived parameters (or none valid). Only save sampled params.
        np.savetxt("full_posterior_samples.txt", posterior_valid, header=",".join(param_names))
    else:
        full_posterior = np.hstack((posterior_valid, derived_array_valid))
        np.savetxt("full_posterior_samples.txt", full_posterior, header=",".join(full_param_names))
        
        
        
        
    
        # [FIX] Reconstruct derived_posterior list-of-dicts for compatibility with printing/plotting functions
    if derived_array_valid.size > 0:
        # Convert values to plain Python float for nicer printing/compatibility
        derived_posterior = [dict(zip(derived_only, [float(v) for v in row])) for row in derived_array_valid]
    else:
        derived_posterior = []
        print(f"WARNING: Derived Posterior is empty  ")
    # For downstream percentile calculations use posterior_valid as the canonical posterior (sampled params)
    posterior = posterior_valid
        
 
        
    
    # Shape check: count rows & columns
    try:
        with open("full_posterior_samples.txt", "r") as f:
            lines = [line for line in f if not line.startswith("#")]  # skip header
            num_samples = len(lines)
            num_columns = len(lines[0].strip().split()) if lines else 0
        print(f"Posterior shape = (samples, total parameters) = ({num_samples}, {num_columns})")
    except Exception as e:
        print(f"⚠️ Could not read posterior file to determine shape: {e}")


    
    
    
    # Compute and print median and uncertainties for each sampled parameter
    best_fit_values = np.percentile(posterior, [16, 50, 84], axis=0)
    print("\nBest-fit values (with 1σ uncertainties) for sampled parameters:")
    for i, param in enumerate(param_names):
        median = best_fit_values[1, i]                         # 50th percentile (median)
        lower = best_fit_values[1, i] - best_fit_values[0, i]  # Difference from 16th percentile
        upper = best_fit_values[2, i] - best_fit_values[1, i]  # Difference from 84th percentile
        print(f"{param}: {median:.10g} (+{upper:.10g}, -{lower:.10g})")
        
        
        
    # Compute percentiles for derived parameters
    if derived_posterior: 
        print("\nBest-fit values (with 1σ uncertainties) for derived parameters:")

        derived_keys = list(derived_posterior[0].keys())
        derived_array = np.array([[d[k] for k in derived_keys] for d in derived_posterior])

        derived_percentiles = np.percentile(derived_array, [16, 50, 84], axis=0)

        for i, param in enumerate(derived_keys):
            median = derived_percentiles[1, i]
            lower = median - derived_percentiles[0, i]
            upper = derived_percentiles[2, i] - median
            print(f"{param}: {median:.10g} (+{upper:.10g}, -{lower:.10g})")
        
    
    
    
    # Generate posterior plots
  #  plot_posterior(posterior, derived_posterior, param_names)
    
    
    
    
    
   
    
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")











