import os
import pandas as pd
import io
import argparse
from sklearn.metrics import mean_absolute_error
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Util.Padding import unpad

def decrypt_file_to_df(file_path):
    """Decrypts a .enc file and returns a pandas DataFrame."""
    # 1. Get Private Key from GitHub Secret
    private_key_str = os.getenv('RSA_PRIVATE_KEY')
    if not private_key_str:
        raise ValueError("RSA_PRIVATE_KEY secret is not set in GitHub.")

    # 2. Prepare the RSA Key
    # Trim potential whitespace from the secret
    private_key = RSA.import_key(private_key_str.strip())
    
    with open(file_path, 'rb') as f:
        # Read the size of the encrypted session key (first 2 bytes)
        enc_session_key_size = int.from_bytes(f.read(2), byteorder='big')
        # Read the encrypted session key
        enc_session_key = f.read(enc_session_key_size)
        # Read the AES initialization vector (16 bytes)
        iv = f.read(16)
        # The rest is the ciphertext
        ciphertext = f.read()

    # 3. Decrypt the AES session key with RSA
    cipher_rsa = PKCS1_OAEP.new(private_key)
    session_key = cipher_rsa.decrypt(enc_session_key)

    # 4. Decrypt the CSV data with AES
    cipher_aes = AES.new(session_key, AES.MODE_CBC, iv)
    decrypted_raw = unpad(cipher_aes.decrypt(ciphertext), AES.block_size)
    
    # 5. Convert bytes to DataFrame
    return pd.read_csv(io.BytesIO(decrypted_raw))

def evaluate(submission_file_path):
    # 1. Load the participant's data (Decrypt if necessary)
    try:
        if submission_file_path.endswith('.enc'):
            submission_df = decrypt_file_to_df(submission_file_path)
        else:
            submission_df = pd.read_csv(submission_file_path)
    except Exception as e:
        print(f"ERROR: Could not process submission file - {e}")
        return

    # 2. Load Ground Truth from GitHub Secret
    secret_data = os.getenv('TEST_LABELS')
    if not secret_data:
        print("ERROR: TEST_LABELS secret is not set in GitHub.")
        return
    
    true_df = pd.read_csv(io.StringIO(secret_data.strip()))

    # 3. Standardize column names (lowercase and strip whitespace)
    true_df.columns = true_df.columns.str.lower().str.strip()
    submission_df.columns = submission_df.columns.str.lower().str.strip()

    # 4. Validation
    required_cols = ['subject_session', 'age_at_visit']
    for col in required_cols:
        if col not in submission_df.columns:
            print(f"ERROR: Missing required column '{col}'")
            return

    # 5. Alignment
    # Ensure ID comparison is clean (no extra spaces)
    true_df['subject_session'] = true_df['subject_session'].astype(str).str.strip()
    submission_df['subject_session'] = submission_df['subject_session'].astype(str).str.strip()

    missing = set(true_df['subject_session']) - set(submission_df['subject_session'])
    if missing:
        print(f"ERROR: Submission is missing {len(missing)} IDs. First few: {list(missing)[:3]}")
        return
    
    # Use an inner join to ensure we only score rows that exist in our ground truth
    merged = pd.merge(true_df, submission_df, on='subject_session', how='inner', suffixes=('_true', '_pred'))

    # 6. Calculate Metric
    y_true = merged['age_at_visit_true']
    y_pred = merged['age_at_visit_pred']

    # Final check for NaNs in predictions
    if y_pred.isnull().values.any():
        print("ERROR: Submission contains NaN values in age_at_visit.")
        return

    mae = mean_absolute_error(y_true, y_pred)
    
    # This print statement is what the Bot regex catches
    print(f"SCORE_MAE: {mae:.8f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help="Path to the predictions file")
    args = parser.parse_args()
    
    evaluate(args.file)