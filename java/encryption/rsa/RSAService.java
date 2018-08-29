package snippet

import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.SecureRandom;
import java.security.interfaces.RSAPublicKey;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.X509EncodedKeySpec;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;

public class RSAService {

    /*
    Implementation of RSA encryption using BouncyCastle / Java native functions

    Using BouncyCastle to decode pem certificate
     */

    private static final String ENCRYPTION_METHOD_NAME = "RSA";
    private static final int KEY_SIZE = 2048;

    private KeyPairGenerator kpg;
    private static PrivateKey privateKey;
    private static PublicKey publicKey;

    public void Initialize()throws NoSuchAlgorithmException {
        kpg = KeyPairGenerator.getInstance(ENCRYPTION_METHOD_NAME);
        kpg.initialize(KEY_SIZE);
    }

    public BigInteger getPublicKeyModulus() {
        RSAPublicKey rsaPub  = (RSAPublicKey)(publicKey);
        return rsaPub.getModulus();
    }

    /*
    Remove certificate headers and decode public key
     */
    public static void ComputePublicKey(String certificate) throws NoSuchAlgorithmException, InvalidKeySpecException {
        //Remove header
        certificate = certificate.replace("-----BEGIN PUBLIC KEY-----\r\n", "");
        certificate = certificate.replace("-----END PUBLIC KEY-----", "");

        byte[] encoded = android.util.Base64.decode(certificate, android.util.Base64.DEFAULT);

        KeyFactory kf = KeyFactory.getInstance("RSA");
        publicKey = kf.generatePublic(new X509EncodedKeySpec(encoded));
    }

    /*
    Encrypt provided plain text using previously generated public key and RSA PCKS#1 Cipher
     */
    public byte[] RSAEncrypt(final String plain) throws NoSuchAlgorithmException, NoSuchPaddingException,
            InvalidKeyException, IllegalBlockSizeException, BadPaddingException, NoSuchProviderException {
        //RSA without block splitting and with PCKS1 standard padding, using random generator
        //Provider = BouncyCastle
        Cipher cipher = Cipher.getInstance("RSA/None/PKCS1Padding", "BC");
        SecureRandom random = new SecureRandom();

        cipher.init(Cipher.ENCRYPT_MODE, publicKey, random);

        byte[] encryptedBytes = new byte[0];

        try {
            //Enforce UTF8 to know what we are dealing with
            encryptedBytes = cipher.doFinal(plain.getBytes("UTF-8"));
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        return encryptedBytes;
    }

    public String RSADecrypt(final byte[] encryptedBytes) throws NoSuchAlgorithmException, NoSuchPaddingException,
            InvalidKeyException, IllegalBlockSizeException, BadPaddingException, NoSuchProviderException {
        Cipher cipher = Cipher.getInstance("RSA/None/PKCS1Padding", "BC");
        cipher.init(Cipher.DECRYPT_MODE, privateKey);

        byte[] decryptedBytes = cipher.doFinal(encryptedBytes);
        String decrypted = new String(decryptedBytes);

        return decrypted;
    }
}
