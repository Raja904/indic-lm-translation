"""
test_tokenizer.py — Sanity check for the trained SentencePiece tokenizer.
"""

import sentencepiece as spm

def test():
    model_path = "tokenizer/spm_hi_mr.model"
    sp = spm.SentencePieceProcessor(model_file=model_path)

    samples = [
        "नमस्ते, आप कैसे हैं?", # Hindi
        "नमस्कार, तुम्ही कसे आहात?", # Marathi
        "यह एक परीक्षण वाक्य है।", # Hindi
        "हे एक चाचणी वाक्य आहे." # Marathi
    ]

    print(f"{'='*60}")
    print(f"Testing Tokenizer: {model_path}")
    print(f"Vocab Size: {sp.get_piece_size()}")
    print(f"{'='*60}\n")

    for text in samples:
        # Encode
        tokens = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        
        # Decode
        decoded = sp.decode(ids)
        
        print(f"Original : {text}")
        print(f"Tokens   : {tokens}")
        print(f"IDs      : {ids}")
        print(f"Decoded  : {decoded}")
        print(f"Match?   : {text == decoded}")
        print("-" * 40)

if __name__ == "__main__":
    test()
