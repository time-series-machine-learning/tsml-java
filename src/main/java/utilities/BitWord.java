package utilities;

import java.io.Serializable;
import java.util.Arrays;


/**
 * Provides a simple skeleton implementation for bit packed words, as a replacement for Java Strings.
 * 
 * Currently only supports alphabet size <= 4, wordlength <= 16 
 * 
 * As of 2016/03/05 only incorporated into SFA/BOSS, TODO implement into SAX
 * 
 * @author James Large
 */
public class BitWord implements Comparable<BitWord>, Serializable { 

    public enum PrintFormat {
        RAW,            //simple decimal integer value
        BINARY,         //as 32 bit binary string 
        LETTERS,        //as char string, has unpacking cost
        STRING;         //as string of actual characters
    }
    
    private static final String[] alphabet = { "a","b","c","d","e","f","g","h","i","j" };
    
    private static String[] alphabetSymbols = { "a","b","c","d" };
    
    private int word;
    private byte length;
    
    //ALPHABETSIZE CURRENTLY ASSUMED TO BE 4, THEREFORE 2 BITS PER LETTER, AND MAX WORD LENGTH 16
    public static final int WORD_SPACE = 32; //length of int
    public static final int BITS_PER_LETTER = 2;
    public static final int MAX_LENGTH = WORD_SPACE/BITS_PER_LETTER;
    
    //masks
    private static final int POP_MASK = 0b11;
    private static final int [] LETTER_MASKS = { //again, assumes alphabetsize = 4 still
        0b00000000000000000000000000000011,
        0b00000000000000000000000000001100,
        0b00000000000000000000000000110000,
        0b00000000000000000000000011000000,
        0b00000000000000000000001100000000,
        0b00000000000000000000110000000000,
        0b00000000000000000011000000000000,
        0b00000000000000001100000000000000,
        0b00000000000000110000000000000000,
        0b00000000000011000000000000000000,
        0b00000000001100000000000000000000,
        0b00000000110000000000000000000000,
        0b00000011000000000000000000000000,
        0b00001100000000000000000000000000,
        0b00110000000000000000000000000000,
        0b11000000000000000000000000000000
    };
    
    public BitWord() {
        word = 0;
        length = 32;
    }
    
    public BitWord(BitWord bw) {
        this.word = bw.word;
        this.length = bw.length;
    }
     
    public BitWord(int length) {//throws Exception {
//        if (length > MAX_LENGTH)
//            throw new Exception("requested word length exceeds max(" + MAX_LENGTH + "): " + length);
        
        word = 0;
        length = length;
    }
    
    
    public BitWord(int [] letters) throws Exception {
        setWord(letters);
    }
    
    public int getWord() { return word; }
    public int getLength() { return length; }
    
    public void setWord(int [] letters) {// throws Exception {
//         if (letters.length > MAX_LENGTH)
//            throw new Exception("requested word length exceeds max(" + MAX_LENGTH + "): " + letters.length);
         
         word = 0;
         length = (byte)letters.length;
         
         packAll(letters);
    }
    
    public void push(int letter) {
        word = (word << BITS_PER_LETTER) | letter;
        ++length;
    }
    
    public int pop() {
        int letter = word & POP_MASK;
        shorten(1);
        return letter;
    }
    
    public void packAll(int [] letters) {
        for (int i = 0; i < letters.length; ++i)
            push(letters[i]);
    }
    
    public int[] unpackAll() {
        int [] letters = new int[length];
        
        int shift = WORD_SPACE-(length*BITS_PER_LETTER); //first shift, increment latter
        for (int i = length-1; i > -1; --i) {    
            //left shift to left end to remove earlier letters, right shift to right end to remove latter
            letters[length-1-i] = (int)(word << shift) >>> (WORD_SPACE-BITS_PER_LETTER);
            
            shift += BITS_PER_LETTER;
        }
        
        
            
        return letters;
    }
    
    public void shortenByFourierCoefficient() {
        shorten(2); //1 real/imag pair
    }
    
    public void shorten(int amount) {
        length -= amount;
        word >>>= amount*BITS_PER_LETTER;
    }
    
    @Override
    public int compareTo(BitWord o) {
        return Integer.compare(word, o.word);
    }
    
    @Override
    public boolean equals(Object other) {
        if (other instanceof BitWord)
            return compareTo((BitWord)other) == 0;
        return false;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 29 * hash + this.word;
        hash = 29 * hash + this.length;
        return hash;
    }
    
    public String buildString() {
        int [] letters = unpackAll();
        StringBuilder word = new StringBuilder();
        for (int i = 0; i < letters.length; ++i)
            word.append(alphabet[letters[i]]);
        return word.toString();
    }
    
    @Override
    public String toString() {
        return Arrays.toString(unpackAll());
    }
    public String toString(PrintFormat format) {
        switch (format) {
            case RAW:
                return String.valueOf(Integer.toString(word));
            case BINARY: 
                return String.format("%"+WORD_SPACE+"s", Integer.toBinaryString(word)).replace(' ', '0');
            case LETTERS: {
                return Arrays.toString(unpackAll());
            }
            case STRING: {
                return buildString();
            }
            default:
                return "err"; //impossible with enum, but must have return
        }
    }
    
    public static void main(String [] args) throws Exception {
        quickTest();
        //buildMasks();

        
    }
    
    private static void buildMasks() {
        for (int i = 0; i < 16; ++i) {
            System.out.print("0b");
            for (int j = 15; j > i; --j)
                System.out.print("00");
            System.out.print("11");
            for (int j = 0; j < i; ++ j)
                System.out.print("00");
            System.out.println(",");
        }
    }
    
    private static void quickTest() throws Exception { 
        int [] letters = {0,1,2,3,2,1,2,0,1};
        BitWord w = new BitWord(letters);
        System.out.println(Arrays.toString(letters));
//        System.out.println(w);
//        System.out.println(w.toString(PrintFormat.RAW));
//        System.out.println(w.toString(PrintFormat.BINARY));
        System.out.println(w.toString(PrintFormat.LETTERS));
        w.shortenByFourierCoefficient();
        System.out.println(w.toString(PrintFormat.LETTERS));
        
        
        System.out.println("  ");
        
        
        int [] letters2 = {0,1,2,3,2,1,2,0};
        BitWord w2 = new BitWord(letters2);
        System.out.println(Arrays.toString(letters2));
//        System.out.println(w2);
//        System.out.println(w2.toString(PrintFormat.RAW));
//        System.out.println(w2.toString(PrintFormat.BINARY));
        System.out.println(w2.toString(PrintFormat.LETTERS));
        w2.shortenByFourierCoefficient();
        System.out.println(w2.toString(PrintFormat.LETTERS));
    }
}
