using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Org.BouncyCastle.Crypto;
using Org.BouncyCastle.Crypto.Encodings;
using Org.BouncyCastle.Crypto.Engines;
using Org.BouncyCastle.Crypto.Generators;
using Org.BouncyCastle.Crypto.Prng;
using Org.BouncyCastle.OpenSsl;
using Org.BouncyCastle.Security;

namespace Services
{
    public class RSAService
    {
        private const int KEY_SIZE = 2048;

        public static AsymmetricCipherKeyPair AsymmetricKeys;

        #region BouncyCastle
        public void BC_GenerateKeys()
        {
            CryptoApiRandomGenerator randomGenerator = new CryptoApiRandomGenerator();
            SecureRandom secureRandom = new SecureRandom(randomGenerator);
            var keyGenerationParameters = new KeyGenerationParameters(secureRandom, KEY_SIZE);

            var keyPairGenerator = new RsaKeyPairGenerator();
            keyPairGenerator.Init(keyGenerationParameters);

            AsymmetricKeys = keyPairGenerator.GenerateKeyPair();
        }

        public string BC_GeteyStringRepresentation(AsymmetricKeyParameter key)
        {
            TextWriter textWriter = new StringWriter();
            PemWriter pemWriter = new PemWriter(textWriter);
            pemWriter.WriteObject(key);
            pemWriter.Writer.Flush();

            return textWriter.ToString();
        }

        public string BC_RsaEncryptWithPublic(string clearText)
        {
            var bytesToEncrypt = Encoding.UTF8.GetBytes(clearText);
            var encryptEngine = new Pkcs1Encoding(new RsaEngine());

            encryptEngine.Init(true, AsymmetricKeys.Public);

            var encrypted = Convert.ToBase64String(encryptEngine.ProcessBlock(bytesToEncrypt, 0, bytesToEncrypt.Length));

            return encrypted;
        }

        public string BC_RsaDecryptWithPrivate(string base64Input)
        {
            var bytesToDecrypt = Convert.FromBase64String(base64Input);
            var decryptEngine = new Pkcs1Encoding(new RsaEngine());

            decryptEngine.Init(false, AsymmetricKeys.Private);            

            var decrypted = Encoding.UTF8.GetString(decryptEngine.ProcessBlock(bytesToDecrypt, 0, bytesToDecrypt.Length));

            return decrypted;
        }

        public void Initialize()
        {
            BC_GenerateKeys();
        }

        #endregion
    }
}
