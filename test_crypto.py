
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

def pad(data: bytes, block_size: int = AES.block_size) -> bytes:
    """PKCS7 padding."""
    padding_len = block_size - len(data) % block_size
    return data + bytes([padding_len]) * padding_len

def unpad(data: bytes) -> bytes:
    """Remove PKCS7 padding."""
    padding_len = data[-1]
    return data[:-padding_len]

# 1. Generate a random 256-bit (32-byte) key
key = get_random_bytes(32)
print(f"KEY: {key}")

# 2. Your plaintext
plaintext = b"Secret message goes here"

# 3. Generate a random IV
EAS_block_size=AES.block_size
iv = get_random_bytes(AES.block_size)

# 4. Encrypt
cipher = AES.new(key, AES.MODE_CBC, iv)
ciphertext = cipher.encrypt(pad(plaintext))

# 5. Encode key, IV, and ciphertext in Base64 for storage or transport
b64_key = base64.b64encode(key).decode()
b64_iv = base64.b64encode(iv).decode()
b64_ct = base64.b64encode(ciphertext).decode()

print("AES-256-CBC Encryption")
print("Key:        ", b64_key)
print("IV:         ", b64_iv)
print("Ciphertext: ", b64_ct)

# 6. Decrypt to verify
dec_cipher = AES.new(base64.b64decode(b64_key), AES.MODE_CBC, base64.b64decode(b64_iv))
decrypted = unpad(dec_cipher.decrypt(base64.b64decode(b64_ct)))
print("Decrypted:  ", decrypted.decode())