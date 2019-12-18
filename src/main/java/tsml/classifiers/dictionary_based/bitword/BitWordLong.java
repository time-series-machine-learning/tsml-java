/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers.dictionary_based.bitword;

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
public class BitWordLong implements Comparable<BitWordLong>, Serializable {

    protected static final long serialVersionUID = 22573L;

    public enum PrintFormat {
        RAW,            //simple decimal integer value
        BINARY,         //as 32 bit binary string
        LETTERS,        //as char string, has unpacking cost
        STRING;         //as string of actual characters
    }

    protected static final String[] alphabet = { "a","b","c","d","e","f","g","h","i","j" };

    protected static String[] alphabetSymbols = { "a","b","c","d" };

    protected long word;
    //protected byte length;

    //masks
    protected static final int POP_MASK = 0b11;
    protected static final int [] LETTER_MASKS = { //again, assumes alphabetsize = 4 still
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

    //ALPHABETSIZE CURRENTLY ASSUMED TO BE 4, THEREFORE 2 BITS PER LETTER, AND MAX WORD LENGTH 16
    public static final int WORD_SPACE = 64; //length of long
    public static final int BITS_PER_LETTER = 2;
    public static final int MAX_LENGTH = 32; //WORD_SPACE/BITS_PER_LETTER;

    public BitWordLong() {
        word = 0L;
        //length = 32;
    }

    public BitWordLong(BitWordLong bw) {
        this.word = bw.word;
        //this.length = bw.length;
    }

    public BitWordLong(int length) {//throws Exception {
//        if (length > MAX_LENGTH)
//            throw new Exception("requested word length exceeds max(" + MAX_LENGTH + "): " + length);

        word = 0L;
        //length = length;
    }


    public BitWordLong(int [] letters) throws Exception {
        setWord(letters);
    }

    public BitWordLong(BitWordLong bw1, BitWordLong bw2){
        word = (bw1.word << 32) | bw2.word;
        //length = 32;
    }
    
    public void setWord(int [] letters) {// throws Exception {
//         if (letters.length > MAX_LENGTH)
//            throw new Exception("requested word length exceeds max(" + MAX_LENGTH + "): " + letters.length);
         
         word = 0;
         //length = (byte)letters.length;
         
         packAll(letters);
    }
    
    public void push(int letter) {
        word = ((Long)word << BITS_PER_LETTER) | letter;
        //++length;
    }
    
    public long pop() {
        long letter = (Long)word & POP_MASK;
        shorten(1);
        return letter;
    }
    
    public void packAll(int [] letters) {
        for (int i = 0; i < letters.length; ++i)
            push(letters[i]);
    }
    
//    public int[] unpackAll() {
//        int [] letters = new int[length];
//
//        int shift = WORD_SPACE-(length*BITS_PER_LETTER); //first shift, increment latter
//        for (int i = length-1; i > -1; --i) {
//            //left shift to left end to remove earlier letters, right shift to right end to remove latter
//            letters[length-1-i] = (int)((Long)word << shift) >>> (WORD_SPACE-BITS_PER_LETTER);
//
//            shift += BITS_PER_LETTER;
//        }
//
//
//
//        return letters;
//    }
    
    public void shorten(int amount) {
        //length -= amount;
        word = word >>> amount*BITS_PER_LETTER;
    }

    @Override
    public int compareTo(BitWordLong o) {
        return Long.compare(word, o.word);
    }

    public void shortenByFourierCoefficient() {
        shorten(2); //1 real/imag pair
    }

    public long getWord() { return word; }
    //public int getLength() { return length; }
    
    @Override
    public boolean equals(Object other) {
        if (other instanceof BitWordLong)
            return compareTo((BitWordLong)other) == 0;
        return false;
    }

    @Override
    public int hashCode() {
//        int hash = 7;
//        hash = 29 * hash + Long.hashCode((Long)this.word);
//        hash = 29 * hash + this.length;
//        return hash;
        return Long.hashCode(word);
    }
    
//    public String buildString() {
//        int [] letters = unpackAll();
//        StringBuilder word = new StringBuilder();
//        for (int i = 0; i < letters.length; ++i)
//            word.append(alphabet[letters[i]]);
//        return word.toString();
//    }
    
    @Override
    public String toString() {
        return toString(PrintFormat.BINARY);
    }
    public String toString(PrintFormat format) {
        switch (format) {
            case RAW:
                return String.valueOf(Long.toString((Long)word));
            case BINARY: 
                return String.format("%"+WORD_SPACE+"s", Long.toBinaryString((Long)word)).replace(' ', '0');
            case LETTERS: {
                //return Arrays.toString(unpackAll());
            }
            case STRING: {
                //return buildString();
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
        int [] letters = {2,1,2,3,2,1,2,0,1,1,2,3,1,2,3,1};
        BitWordLong b = new BitWordLong();
        for (int i = 0; i < letters.length; i++){
            b.push(letters[i]);
            System.out.println(b.toString(BitWordLong.PrintFormat.BINARY));
        }

//        b.shorten(6);
//        System.out.println(b.toString(BitWordLong.PrintFormat.BINARY));

        b.word = ((Long)b.word << 32) | (Long)b.word;
        System.out.println(b.toString(BitWordLong.PrintFormat.BINARY));

        System.out.println();

        BitWordInt b2 = new BitWordInt();
        for (int i = 0; i < letters.length; i++){
            b2.push(letters[i]);
            System.out.println(b2.toString(BitWordInt.PrintFormat.BINARY));
        }

        b2.shorten(6);
        System.out.println(b2.toString(BitWordInt.PrintFormat.BINARY));

        System.out.println(Long.hashCode(10000L));
        System.out.println(Long.hashCode(10000000000001L));
        System.out.println(Long.hashCode(1316137241L));
    }
}
