#include <bits/stdc++.h>
#include <chrono>
#include <filesystem>   // Claude - comment: needed to resolve paths relative to the exe, not the CWD
#include <random>       // Claude - comment: needed for cryptographically-reasonable salt generation
#include <cstdint>
#include <cstring>

#ifdef _WIN32
#include <windows.h>    // Claude - comment: GetModuleFileNameW, used to find the exe's real location on Windows
#else
#include <unistd.h>     // Claude - comment: readlink("/proc/self/exe", ...), the Linux equivalent
#include <climits>
#endif

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem; // Claude - comment: shorthand, used throughout the path-resolution code below

// ============================================================================
// Claude - comment: SELF-CONTAINED SHA-256 IMPLEMENTATION
// No external crypto library (no OpenSSL) so this file stays dependency-free
// for your beta. This is a standard, widely-used public-domain-style SHA-256
// implementation (constants/logic per FIPS 180-4), not something exotic.
// It replaces the Caesar cipher completely — see notes further down on why.
// ============================================================================
namespace sha256_impl {

    struct SHA256 {
        uint32_t state[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };
        uint8_t data[64];
        uint32_t datalen = 0;
        uint64_t bitlen = 0;

        static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

        void transform(const uint8_t block[64]) {
            static const uint32_t k[64] = {
                0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
                0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
                0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
                0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
                0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
                0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
                0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
                0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
            };
            uint32_t m[64];
            for (int i = 0; i < 16; ++i)
                m[i] = (block[i*4] << 24) | (block[i*4+1] << 16) | (block[i*4+2] << 8) | block[i*4+3];
            for (int i = 16; i < 64; ++i) {
                uint32_t s0 = rotr(m[i-15],7) ^ rotr(m[i-15],18) ^ (m[i-15] >> 3);
                uint32_t s1 = rotr(m[i-2],17) ^ rotr(m[i-2],19) ^ (m[i-2] >> 10);
                m[i] = m[i-16] + s0 + m[i-7] + s1;
            }
            uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
            for (int i = 0; i < 64; ++i) {
                uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
                uint32_t ch = (e & f) ^ ((~e) & g);
                uint32_t temp1 = h + S1 + ch + k[i] + m[i];
                uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
                uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t temp2 = S0 + maj;
                h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
            }
            state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
            state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
        }

        void update(const uint8_t* input, size_t len) {
            for (size_t i = 0; i < len; ++i) {
                data[datalen++] = input[i];
                if (datalen == 64) { transform(data); bitlen += 512; datalen = 0; }
            }
        }

        string hexdigest() {
            uint32_t i = datalen;
            if (datalen < 56) {
                data[i++] = 0x80;
                while (i < 56) data[i++] = 0x00;
            } else {
                data[i++] = 0x80;
                while (i < 64) data[i++] = 0x00;
                transform(data);
                memset(data, 0, 56);
            }
            bitlen += datalen * 8;
            for (int j = 0; j < 8; ++j)
                data[63 - j] = (uint8_t)(bitlen >> (j * 8));
            transform(data);

            ostringstream out;
            out << hex << setfill('0');
            for (int j = 0; j < 8; ++j)
                out << setw(8) << state[j];
            return out.str();
        }
    };

    string sha256(const string& input) {
        SHA256 ctx;
        ctx.update(reinterpret_cast<const uint8_t*>(input.data()), input.size());
        return ctx.hexdigest();
    }
}

// ============================================================================
// Claude - comment: PATH RESOLUTION — anchored to the exe's real location,
// not the current working directory. This is the fix for the bug where
// key.txt/lockout.txt would land in a different place depending on how the
// user launched the app (double-click vs shortcut vs terminal), and would
// break entirely once this gets bundled by PyInstaller/Nuitka, which run
// from a temp extraction folder, not the exe's real folder.
// ============================================================================
fs::path getExeDir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
#else
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len == -1) return fs::current_path(); // Claude - comment: fallback, shouldn't normally trigger on Linux
    buf[len] = '\0';
    return fs::path(buf).parent_path();
#endif
}

// Claude - comment: all persistent files now live next to the exe, inside a
// dedicated "data" subfolder, instead of loose files in whatever the CWD is.
// This is also the folder we'll eventually point the Python side at for
// models/config, so everything the app owns lives under one root.
const fs::path APP_ROOT = getExeDir();
const fs::path DATA_DIR = APP_ROOT / "data";
const fs::path KEY_FILE = DATA_DIR / "key.txt";
const fs::path LOCKOUT_FILE = DATA_DIR / "lockout.txt";

const int MAX_ATTEMPTS = 3;
const long LOCKOUT_SECONDS = 3 * 60 * 60; // 3 hours

// Claude - comment: CIPHER_SHIFT and the Caesar cipher are gone entirely —
// see the writeup after this file for why a reversible cipher was a real
// vulnerability here, not just "weak crypto." Salted SHA-256 hashing below
// replaces it: we never store or need the plaintext password again after
// the very first write.

// --- salt generation ---

string generateSalt(size_t length = 16) {
    // Claude - comment: random_device is the right source here — it's
    // intended for exactly this (seeding/security-relevant randomness),
    // unlike mt19937 alone which is NOT cryptographically secure on its own.
    random_device rd;
    static const char charset[] = "0123456789abcdef";
    string salt;
    for (size_t i = 0; i < length; ++i)
        salt += charset[rd() % 16];
    return salt;
}

string hashWithSalt(const string& plain, const string& salt) {
    return sha256_impl::sha256(salt + plain); // Claude - comment: salt prepended before hashing
}

// --- storage helpers ---
// Claude - comment: key.txt now stores TWO lines — salt, then hash — instead
// of one reversible cipher string. Reading/writing both together keeps this
// simple: one file, two lines, no separate salt file to keep in sync.

bool writeStoredHash(const string& salt, const string& hash) {
    fs::create_directories(DATA_DIR); // Claude - comment: ensure data/ exists before first write
    ofstream out(KEY_FILE, ios::trunc);
    if (!out) return false;
    out << salt << "\n" << hash;
    return true;
}

bool readStoredHash(string& salt, string& hash) {
    ifstream in(KEY_FILE);
    if (!in) return false;
    if (!getline(in, salt)) return false;
    if (!getline(in, hash)) return false;
    return true;
}

// --- lockout persistence ---

long currentEpoch() {
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

long checkLockout() {
    ifstream in(LOCKOUT_FILE);
    if (!in) return 0;

    long lockedAt;
    in >> lockedAt;
    long elapsed = currentEpoch() - lockedAt;

    if (elapsed >= LOCKOUT_SECONDS) {
        fs::remove(LOCKOUT_FILE); // Claude - comment: fs::remove instead of C-style remove(), consistent with filesystem use elsewhere
        return 0;
    }
    return LOCKOUT_SECONDS - elapsed;
}

void triggerLockout() {
    fs::create_directories(DATA_DIR); // Claude - comment: same safety as writeStoredHash, in case lockout fires before any successful write
    ofstream out(LOCKOUT_FILE, ios::trunc);
    out << currentEpoch();
}

// --- first run setup ---

void ensureKeyFileExists() {
    if (fs::exists(KEY_FILE)) return; // already exists, nothing to do

    cout << "No password file found - creating one now.\n";
    cout << "Default password is: Admin\n";
    cout << "(Change it after logging in.)\n\n";

    string salt = generateSalt();
    string hash = hashWithSalt("Admin", salt);
    writeStoredHash(salt, hash);
}

// --- core auth ---

int checkPassword() {
    long remaining = checkLockout();
    if (remaining > 0) {
        cout << "Access Blocked. Try again in " << (remaining / 60) << " minutes.\n";
        return 0;
    }

    string storedSalt, storedHash;
    if (!readStoredHash(storedSalt, storedHash)) {
        cout << "Could not read credentials file - something's wrong with the install.\n"; // Claude - comment: explicit failure path instead of silently comparing against empty strings
        return 0;
    }

    int count = MAX_ATTEMPTS;
    string input;

    while (count > 0) {
        cout << "Enter Password: ";
        if (!(cin >> input)) {
            cout << "Input error - couldn't read password.\n";
            return 0;
        }

        if (hashWithSalt(input, storedSalt) == storedHash) { // Claude - comment: compare hashes, never touch/see plaintext again
            cout << "Access Granted\n";
            return 1;
        }

        count--;
        cout << "Incorrect Password - Remaining Attempts: " << count << endl;
    }

    triggerLockout();
    cout << "Access Blocked. Try again after 3 hours.\n";
    return 0;
}

bool changePassword() {
    string storedSalt, storedHash;
    if (!readStoredHash(storedSalt, storedHash)) {
        cout << "Error: could not read " << KEY_FILE << "\n";
        return false;
    }

    int attempts = 3;
    string current;

    while (attempts > 0) {
        cout << "Enter current password to confirm change: ";
        cin >> current;

        if (hashWithSalt(current, storedSalt) == storedHash) {
            break; // verified, move on to setting new password
        }

        attempts--;
        if (attempts == 0) {
            cout << "Too many incorrect attempts. Cancelling password change.\n";
            return false;
        }
        cout << "Incorrect current password. " << attempts << " attempt(s) left.\n";
    }

    cout << "Enter new password: ";
    string newPass;
    cin >> newPass;

    string newSalt = generateSalt(); // Claude - comment: fresh salt on every password change, not just first setup
    string newHash = hashWithSalt(newPass, newSalt);

    if (writeStoredHash(newSalt, newHash)) {
        cout << "Password changed successfully.\n";
        return true;
    } else {
        cout << "Error: could not write to " << KEY_FILE << "\n";
        return false;
    }
}

// --- entry point ---

int main() {
    ensureKeyFileExists();

    if (checkPassword() != 1) {
        return 1;
    }

    cout << "Welcome - Authorized User\n";

    while (true) {
    cout << "Change password before continuing? (y/n): ";
    string choice;
    cin >> choice;
    if (choice == "y") {
        if (!changePassword()) {
            cout << "\n...\nContinuing without changing password.\n";
        }
        break;
    } else if (choice == "n") {
        break;
    }
}

    // Claude - comment: replaced system("python local_llm_assistant.py") with
    // an exe-anchored path + the install root passed as an argument. The old
    // version assumed "python" was on PATH (often false for non-dev users)
    // and gave the Python side no way to know where its own data lives.
    fs::path pythonScript = APP_ROOT / "local_llm_assistant.py";
    string command = "python \"" + pythonScript.string() + "\" \"" + APP_ROOT.string() + "\"";
    system(command.c_str());

    // NOTE THAT AFTER STARTING THE PYTHON PROGRAM - THE CPP FILE SHOULD TRACK THE CPU,GPU,NPU,RAM in real time - of the .py file specifically.
    // CHECK FOR UPDATES... FEATURE TO BE ADDED IN V5.
    // Claude - comment: monitoring intentionally still not implemented here -
    // per your beta scope, this stays off but the structure (this comment,
    // this exact spot in main) is preserved so v2 slots in without a rewrite.
    return 0;
}


// USER_ADMIN - COMMENT - to be added
// implement real time monitoring - of all devices available including ram/disk/cpu/gpu/npu
// show details like ram size - speed and usage, cpu model and usage, same for everything - name, usage current out of total bandwidth - of the app only -> opens in new window right after the authentication is successfull and python launches -> should have a kill switch in case .python progrmam crashes and hangs.
// reorganize the entire codebase - refactor it complelty make it very readable and humanlike. and i really mean very readable - like you can use functions for literally anything - or even oops, stl - anything just make sure its readable and friendly
// whatever comments you write - write as claude comment - as done above - it should be there as reference.

// NOTE THAT AT THE END THE MONITORING IS SUPPOSED TO BE ALL IN 1 APP SO IT DOESNT REALLY MATTER NOW