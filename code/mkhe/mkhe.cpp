// @file  threshold-fhe.cpp - Examples of threshold FHE for BGVrns, BFVrns, and
// CKKS
// @author TPOC: contact@palisade-crypto.org
//
// @copyright Copyright (c) 2020, Duality Technologies Inc.
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. THIS SOFTWARE IS
// PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "palisade.h"
#include <random>
#include <chrono>
#include <iomanip>
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "scheme/ckks/ckks-ser.h"
#include "pubkeylp-ser.h"
#include <tuple>
#include <unistd.h>


using namespace std;
using namespace lbcrypto;

template <typename Element>
class CryptoContextFactory;

template <typename Element>
class CryptoContextImpl;

template <typename Element>
using CryptoContext = shared_ptr<CryptoContextImpl<Element>>;

void RunSingleKeyCKKS(int model_size);
void RunCKKS(int model_size, int client_size);

int main(int argc, char *argv[]) {
        // Using std::stoi (string to integer)

int model_size = std::stoi(argv[1]);
std::cout << "Size of the model: " << model_size << std::endl;

int client_size = std::stoi(argv[2]);
std::cout << "Size of the clients: " << client_size << std::endl;

  std::cout << "\n=================RUNNING FOR Single Key CKKS====================="
            << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  RunSingleKeyCKKS(model_size);

      // Stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print the elapsed time
    std::cout << "Single Key Elapsed time: " << duration.count() << " milliseconds" << std::endl;


  std::cout << "\n=================RUNNING FOR CKKS====================="
            << std::endl;

  RunCKKS(model_size, client_size);

          // Stop the timer
    auto end_time1 = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - end_time);

    // Print the elapsed time
    std::cout << "Threshold Elapsed time: " << duration1.count() << " milliseconds" << std::endl;


  return 0;
}

void RunSingleKeyCKKS(int model_size) {
 // Step 1: Setup CryptoContext
  uint32_t multDepth = 1;

  uint32_t scaleFactorBits = 50;

  uint32_t batchSize = 4096;

  SecurityLevel securityLevel = HEStd_128_classic;

  // The following call creates a CKKS crypto context based on the
  // arguments defined above.
  CryptoContext<DCRTPoly> cc =
      CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
          multDepth, scaleFactorBits, batchSize, securityLevel);

  std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension()
            << std::endl
            << std::endl;

  // Enable the features that you wish to use
  cc->Enable(ENCRYPTION);
  cc->Enable(SHE);

  auto keys = cc->KeyGen();

  cc->EvalMultKeyGen(keys.secretKey);

  cc->EvalAtIndexKeyGen(keys.secretKey, {1, -2});

  // Step 3: Encoding and encryption of inputs

  // Inputs

  // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 10.0); // Range of random numbers (0.0 to 10.0)

  vector<double> x1;
  vector<double> x2;

    for (int i = 0; i < model_size; ++i) {
        double random_number = dis(gen);
        x1.push_back(random_number);
        x2.push_back(random_number);
    }

  // Encoding as plaintexts
  Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
  Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

//   std::cout << "Input x1: " << ptxt1 << std::endl;
//   std::cout << "Input x2: " << ptxt2 << std::endl;

  // Encrypt the encoded vectors
  auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
  auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

      Serial::SerializeToFile("CT1.txt", c1,
                          SerType::BINARY);
    Serial::SerializeToFile("CT2.txt", c2,
                          SerType::BINARY);                          

  // Step 4: Evaluation


  // Homomorphic scalar multiplication
  auto cScalar1 = cc->EvalMult(c1, 0.5);
  auto cScalar2 = cc->EvalMult(c2, 0.5);

    // Homomorphic addition
  auto cAdd = cc->EvalAdd(cScalar1, cScalar2);


  // Step 5: Decryption and output
  Plaintext result;
  // We set the cout precision to 8 decimal digits for a nicer output.
  // If you want to see the error/noise introduced by CKKS, bump it up
  // to 15 and it should become visible.
  std::cout.precision(8);

  // Decrypt the result of addition
  cc->Decrypt(keys.secretKey, cAdd, &result);
  result->SetLength(batchSize);
//   std::cout << "x1 + x2 = " << result;
  std::cout << "Estimated precision in bits: " << result->GetLogPrecision()
            << std::endl;

}


void RunCKKS(int model_size, int client_size) {
//    * @param m Cyclotomic order. Must be a power of 2.
//  * @param init_size Number of co-primes comprising the ciphertext modulus.
//  * 			  It is equal to the desired depth of the computation.
//  * @param dcrtBits Size of each co-prime in bits. Should fit into a
//  * 			 machine word, i.e., less than 64.
//  * @param p Scaling parameter 2^p. p should usually be equal to dcrtBits.
//  * @param relinWin The bit decomposition count used in relinearization.
//  * 			 Use 0 to go with max possible. Use small values (3-4?)
//  * 			 if you need rotations before any multiplications.
//  * @param batchSize The length of the packed vectors to be used with CKKS.
//  * @param mode (e.g., RLWE or OPTIMIZED)
//  * @param ksTech key switching technique to use (e.g., GHS or BV)
//  * @param rsTech rescaling technique to use (e.g., APPROXRESCALE or
//  * EXACTRESCALE)

  usint init_size = 3;
  usint dcrtBits = 51;
  usint batchSize = 4096;

  CryptoContext<DCRTPoly> cc =
      CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
          init_size - 1, dcrtBits, batchSize, HEStd_128_classic,
          0,                    /*ringDimension*/
          APPROXRESCALE, BV, 2, /*numLargeDigits*/
          1,                    /*maxDepth*/
          60,                   /*firstMod*/
          5, OPTIMIZED);

  // enable features that you wish to use
  cc->Enable(ENCRYPTION);
  cc->Enable(SHE);
  cc->Enable(LEVELEDSHE);
  cc->Enable(MULTIPARTY);

  ////////////////////////////////////////////////////////////
  // Set-up of parameters
  ////////////////////////////////////////////////////////////

  // Output the generated parameters
  std::cout << "p = " << cc->GetCryptoParameters()->GetPlaintextModulus()
            << std::endl;
  std::cout
      << "n = "
      << cc->GetCryptoParameters()->GetElementParams()->GetCyclotomicOrder() / 2
      << std::endl;
  std::cout << "log2 q = "
            << log2(cc->GetCryptoParameters()
                        ->GetElementParams()
                        ->GetModulus()
                        .ConvertToDouble())
            << std::endl;



  ////////////////////////////////////////////////////////////
  // Perform Key Generation Operation
  ////////////////////////////////////////////////////////////

  // Initialize Public Key Containers
  vector<LPKeyPair<DCRTPoly>> kp(client_size);
  vector<shared_ptr<std::map<usint, LPEvalKey<Element>>>> evalSumKeys(client_size);
  vector<LPEvalKey<Element>> evalMultKeys(client_size);
  LPEvalKey<Element> evalMidMultKey;
  LPEvalKey<Element> evalMultFinal;


  shared_ptr<std::map<usint, LPEvalKey<Element>>> evalSumKeysJoin;


  LPKeyPair<DCRTPoly> kpMultiparty;

  
  
 std::cout << "Running key generation (used for source data)..." << std::endl;

  // Round 1 (party A)

  std::cout << "Round 1 (party A) started." << std::endl;

  kp[0] = cc->KeyGen();

  // Generate evalmult key part for A
  evalMultKeys[0] = cc->KeySwitchGen(kp[0].secretKey, kp[0].secretKey);

  // Generate evalsum key part for A
  cc->EvalSumKeyGen(kp[0].secretKey);
  evalSumKeys[0] = std::make_shared<std::map<usint, LPEvalKey<DCRTPoly>>>(
      cc->GetEvalSumKeyMap(kp[0].secretKey->GetKeyTag()));

  std::cout << "Round 1 of key generation completed." << std::endl;


  for (int i = 1; i < kp.size(); i++) {
    kp[i] = cc->MultipartyKeyGen(kp[i-1].publicKey);
    
    evalSumKeys[i] = cc->MultiEvalSumKeyGen(kp[i].secretKey, evalSumKeys[i-1],
                                             kp[i].publicKey->GetKeyTag());

    if (i == 1) {
      evalSumKeysJoin = cc->MultiAddEvalSumKeys(evalSumKeys[i-1], evalSumKeys[i],
                                                 kp[i].publicKey->GetKeyTag());
    } else {
      evalSumKeysJoin = cc->MultiAddEvalSumKeys(evalSumKeysJoin, evalSumKeys[i],
                                                 kp[i].publicKey->GetKeyTag());
    }

    evalMultKeys[i] =
      cc->MultiKeySwitchGen(kp[i].secretKey, kp[i].secretKey, evalMultKeys[i-1]);
    if (i==1) {
      evalMidMultKey = cc->MultiAddEvalKeys(evalMultKeys[i-1], evalMultKeys[i],
                                         kp2.publicKey->GetKeyTag());
    } else {
      evalMidMultKey = cc->MultiAddEvalKeys(evalMidMultKey, evalMultKeys[i],
                                         kp2.publicKey->GetKeyTag());
    }
  }
  cc->InsertEvalSumKey(evalSumKeysJoin);

  evalMultFinal = cc->MultiMultEvalKey(evalMidMultKey, kp[0].secretKey,
                                          kp[0].publicKey->GetKeyTag());
  for (int i = 1; i < kp.size(); i++) {
    // b * (a + b + c)
    auto evalMultRound3 = cc->MultiMultEvalKey(evalMidMultKey, kp[i].secretKey,
                                          kp[i].publicKey->GetKeyTag());

    auto evalMultFinal = cc->MultiAddEvalMultKeys(evalMultFinal, evalMultRound3,
                                                evalMidMultKey->GetKeyTag());
  }
  cc->InsertEvalMultKey({evalMultFinal});

  ////////////////////////////////////////////////////////////
  // Encode source data
  ////////////////////////////////////////////////////////////

  // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 10.0); // Range of random numbers (0.0 to 10.0)

  vector<double> x1;
  vector<double> x2;

    for (int i = 0; i < model_size; ++i) {
        double random_number = dis(gen);
        x1.push_back(random_number);
        x2.push_back(random_number);
    }


  Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(x1);
  Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(x2);

  ////////////////////////////////////////////////////////////
  // Encryption
  ////////////////////////////////////////////////////////////

  Ciphertext<DCRTPoly> ciphertext1;
  Ciphertext<DCRTPoly> ciphertext2;

  ciphertext1 = cc->Encrypt(kp[client_size-1].publicKey, plaintext1);
  ciphertext2 = cc->Encrypt(kp[client_size-1].publicKey, plaintext2);

    Serial::SerializeToFile("TCT1.txt", ciphertext1,
                          SerType::BINARY);
    Serial::SerializeToFile("TCT2.txt", ciphertext2,
                          SerType::BINARY);                          



  ////////////////////////////////////////////////////////////
  // EvalAdd Operation on Re-Encrypted Data
  ////////////////////////////////////////////////////////////
  // Homomorphic scalar multiplication
  auto cScalar1 = cc->EvalMult(ciphertext1, 0.5);
  auto cScalar2 = cc->EvalMult(ciphertext2, 0.5);

  
  Ciphertext<DCRTPoly> ciphertextAdd12;

  ciphertextAdd12 = cc->EvalAdd(cScalar1, cScalar1);

//   auto ciphertextMultTemp = cc->EvalMult(ciphertext1, ciphertext2);
//   auto ciphertextMult = cc->ModReduce(ciphertextMultTemp);
// //   auto ciphertextEvalSum = cc->EvalSum(ciphertext2, batchSize);

  ////////////////////////////////////////////////////////////
  // Decryption after Accumulation Operation on Encrypted Data with Multiparty
  ////////////////////////////////////////////////////////////

  Plaintext plaintextAddNew1;
  Plaintext plaintextAddNew2;

  DCRTPoly partialPlaintext1;
  DCRTPoly partialPlaintext2;

  Plaintext plaintextMultipartyNew;

  const shared_ptr<LPCryptoParameters<DCRTPoly>> cryptoParams =
      kp1.secretKey->GetCryptoParameters();
  const shared_ptr<typename DCRTPoly::Params> elementParams =
      cryptoParams->GetElementParams();

  // distributed decryption
  vector<Ciphertext<DCRTPoly>> ciphertextPartial(client_size);
  vector<Ciphertext<DCRTPoly>> partialCiphertextVec;

  for (int i = 0; i < kp.size(); i++) {
    ciphertextPartial[i] =
      cc->MultipartyDecryptLead(kp[i].secretKey, {ciphertextAdd12});
    
    partialCiphertextVec.push_back(ciphertextPartial[i][0]);
  }

  cc->MultipartyDecryptFusion(partialCiphertextVec, &plaintextMultipartyNew);

  plaintextMultipartyNew->SetLength(plaintext1->GetLength());

  std::cout << "Estimated precision in bits: " << plaintextMultipartyNew->GetLogPrecision()
            << std::endl;

//   cout << "\n Original Plaintext: \n" << endl;
//   cout << plaintext1 << endl;
//   cout << plaintext2 << endl;

//   plaintextMultipartyNew->SetLength(plaintext1->GetLength());

//   cout << "\n Resulting Fused Plaintext: \n" << endl;
//   cout << plaintextMultipartyNew << endl;

//   cout << "\n";

//   Plaintext plaintextMultipartyMult;

//   ciphertextPartial1 =
//       cc->MultipartyDecryptLead(kp1.secretKey, {ciphertextMult});

//   ciphertextPartial2 =
//       cc->MultipartyDecryptMain(kp2.secretKey, {ciphertextMult});

//   vector<Ciphertext<DCRTPoly>> partialCiphertextVecMult;
//   partialCiphertextVecMult.push_back(ciphertextPartial1[0]);
//   partialCiphertextVecMult.push_back(ciphertextPartial2[0]);

//   cc->MultipartyDecryptFusion(partialCiphertextVecMult,
//                               &plaintextMultipartyMult);

//   plaintextMultipartyMult->SetLength(plaintext1->GetLength());

// //   cout << "\n Resulting Fused Plaintext after Multiplication of plaintexts 1 "
// //           "and 2: \n"
// //        << endl;
// //   cout << plaintextMultipartyMult << endl;

// //   cout << "\n";

//   Plaintext plaintextMultipartyEvalSum;

//   ciphertextPartial1 =
//       cc->MultipartyDecryptLead(kp1.secretKey, {ciphertextEvalSum});

//   ciphertextPartial2 =
//       cc->MultipartyDecryptMain(kp2.secretKey, {ciphertextEvalSum});

//   vector<Ciphertext<DCRTPoly>> partialCiphertextVecEvalSum;
//   partialCiphertextVecEvalSum.push_back(ciphertextPartial1[0]);
//   partialCiphertextVecEvalSum.push_back(ciphertextPartial2[0]);

//   cc->MultipartyDecryptFusion(partialCiphertextVecEvalSum,
//                               &plaintextMultipartyEvalSum);

//   plaintextMultipartyEvalSum->SetLength(plaintext1->GetLength());

//   cout << "\n Fused result after the Summation of ciphertext 3: "
//           "\n"
//        << endl;
//   cout << plaintextMultipartyEvalSum << endl;
}

