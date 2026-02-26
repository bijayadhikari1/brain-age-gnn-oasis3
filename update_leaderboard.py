import os
import pandas as pd
from io import StringIO
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP

def decrypt_file(encrypted_blob, private_key_str):
    """Decrypts the .enc file content using RSA and AES."""
    try:
        private_key = RSA.import_key(private_key_str)
        cipher_rsa = PKCS1_OAEP.new(private_key)
        
        # Split the blob into its components
        k_len = private_key.size_in_bytes()
        enc_session_key = encrypted_blob[:k_len]
        nonce = encrypted_blob[k_len:k_len+16]
        tag = encrypted_blob[k_len+16:k_len+32]
        ciphertext = encrypted_blob[k_len+32:]
        
        # Decrypt session key and data
        session_key = cipher_rsa.decrypt(enc_session_key)
        cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
        data = cipher_aes.decrypt_and_verify(ciphertext, tag)
        return data.decode('utf-8')
    except Exception as e:
        print(f"Decryption error: {e}")
        return None

def calculate_mae(ground_truth_df, prediction_df):
    """Calculates Mean Absolute Error between ground truth and predictions."""
    # Suffixes handle cases where both files have 'age_at_visit'
    merged = pd.merge(ground_truth_df, prediction_df, on='subject_session', suffixes=('_gt', '_pred'))
    
    if merged.empty:
        print("⚠️ Warning: No matching subject_session found between GT and Prediction.")
        return None
    
    # Try to find the age columns regardless of exact suffixing
    gt_col = 'age_at_visit_gt'
    pred_col = 'age_at_visit_pred'
    
    if gt_col in merged.columns and pred_col in merged.columns:
        mae = (merged[gt_col] - merged[pred_col]).abs().mean()
        return round(float(mae), 8)
    else:
        print(f"⚠️ Warning: Missing age columns. Found: {list(merged.columns)}")
        return None

# --- 1. CONFIGURATION & SECRETS ---
gt_data = os.getenv('TEST_LABELS')
priv_key = os.getenv('RSA_PRIVATE_KEY')

if not gt_data or not priv_key:
    print("Error: Missing TEST_LABELS or RSA_PRIVATE_KEY secrets.")
    exit(1)

gt_df = pd.read_csv(StringIO(gt_data))
gt_df.columns = gt_df.columns.str.strip() 

# --- 2. SCAN & DECRYPT SUBMISSIONS ---
submissions_dir = 'submissions'
leaderboard_data = []

if os.path.exists(submissions_dir):
    for filename in os.listdir(submissions_dir):
        if filename.endswith('.enc'):
            team_name = os.path.splitext(filename)[0]
            file_path = os.path.join(submissions_dir, filename)
            
            try:
                with open(file_path, 'rb') as f:
                    decrypted_csv = decrypt_file(f.read(), priv_key)
                
                if decrypted_csv:
                    pred_df = pd.read_csv(StringIO(decrypted_csv))
                    pred_df.columns = pred_df.columns.str.strip()
                    score = calculate_mae(gt_df, pred_df)
                    if score is not None:
                        print(f"✅ Scored {team_name}: {score}")
                        leaderboard_data.append({"TEAM": team_name, "MAE": score})
                else:
                    print(f"⚠️ Skipping {filename}: Decryption failed.")
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

# --- 3. RANKING & DEDUPLICATION ---
if leaderboard_data:
    df = pd.DataFrame(leaderboard_data)
    
    df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')
    df = df.dropna(subset=['MAE', 'TEAM'])

    # Sort by MAE (best score first) then keep only the best entry per team
    df = df.sort_values(by=['MAE']).drop_duplicates(subset=['TEAM'], keep='first')

    # Final Rank Calculation
    df = df.sort_values(by=["MAE", "TEAM"])
    df['RANK'] = df['MAE'].rank(method='dense').astype(int)
    
    leaderboard_df = df[['RANK', 'TEAM', 'MAE']]
    leaderboard_df.columns = ['Rank', 'Team', 'MAE']

    # --- 4. OUTPUT GENERATION ---
    os.makedirs('leaderboard', exist_ok=True)
    os.makedirs('docs', exist_ok=True)

    # Save CSV & Markdown
    leaderboard_df.to_csv('leaderboard/leaderboard.csv', index=False)
    with open('leaderboard/LEADERBOARD.md', 'w') as f:
        f.write("# 🏆 Full Competition History\n\n" + leaderboard_df.to_markdown(index=False))

    # Generate HTML
    html_table = leaderboard_df.to_html(
        classes='table table-hover text-center', 
        index=False,
        formatters={'MAE': lambda x: f"{x:.8f}"}
    )

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
        <title>OASIS3 Challenge</title>
        <style>
            body {{ background-color: #f4f7f6; font-family: 'Inter', sans-serif; padding: 40px 0; }}
            .leaderboard-card {{ background: white; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden; max-width: 800px; margin: auto; }}
            .header-section {{ background: linear-gradient(135deg, #0f172a 0%, #334155 100%); color: white; padding: 40px 20px; }}
            table {{ width: 100% !important; margin-bottom: 0 !important; }}
            th {{ background-color: #f8fafc !important; color: #64748b; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; text-align: center; padding: 15px !important; }}
            td {{ vertical-align: middle; font-size: 1rem; padding: 15px !important; text-align: center; font-weight: 500; }}
            .rank-col {{ font-weight: 700; color: #334155; }}
            .mae-col {{ font-family: monospace; color: #059669; font-weight: 700; }}
        </style>
    </head>
    <body>
        <div class="leaderboard-card">
            <div class="header-section text-center">
                <h1 class="fw-bold">🧠 Brain-Age Prediction Challenge Leaderboard</h1>
                <div class="badge bg-primary mt-2">Last Updated: {pd.Timestamp.now().strftime('%b %d, %H:%M UTC')}</div>
            </div>
            <div class="table-responsive">
                {html_table}
            </div>
        </div>
    </body>
    </html>
    """
    with open('docs/leaderboard.html', 'w') as f:
        f.write(html_content)
    
    print("Leaderboard and HTML updated successfully.")
else:
    print("No valid submission files found to process.")