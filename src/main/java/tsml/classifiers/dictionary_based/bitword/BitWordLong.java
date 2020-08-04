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
 * Used by TDE, stripped down version of original BitWord class using a long to store the word.
 * 
 * @author James Large, updated by Matthew Middlehurst
 */
public class BitWordLong implements BitWord {

    protected static final long serialVersionUID = 1L;

    public enum PrintFormat {
        RAW,            //simple decimal integer value
        BINARY,         //as 32 bit binary string
        LETTERS,        //as char string, has unpacking cost
        STRING;         //as string of actual characters
    }

    private static final String[] alphabet = { "a","b","c","d","e","f","g","h","i","j" };

    private static String[] alphabetSymbols = { "a","b","c","d" };

    private long word;
    private byte length;

    //ALPHABETSIZE CURRENTLY ASSUMED TO BE 4, THEREFORE 2 BITS PER LETTER, AND MAX WORD LENGTH 16
    public static final int WORD_SPACE = 64; //length of long
    public static final int BITS_PER_LETTER = 2;
    public static final int MAX_LENGTH = 32; //WORD_SPACE/BITS_PER_LETTER;

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

    public BitWordLong() {
        word = 0L;
        length = 0;
    }

    public BitWordLong(BitWord bw) {
        this.word = bw.getWord().longValue();
        this.length = bw.getLength();
    }

    public BitWordLong(BitWordLong bw) {
        this.word = bw.word;
        this.length = bw.length;
    }

    public BitWordLong(int [] letters) throws Exception {
        setWord(letters);
    }

    public BitWordLong(BitWord bw1, BitWord bw2){
        if (bw2 instanceof BitWordInt)
            word = (bw1.getWord().longValue() << 32) | Integer.toUnsignedLong(bw2.getWord().intValue());
        else
            word = (bw1.getWord().longValue() << 32) | bw2.getWord().longValue();

        length = 32;
    }

    public Number getWord() { return word; }
    public byte getLength() { return length; }

    public void setWord(Number word) { this.word = word.longValue(); }

    public void setWord(int [] letters) throws Exception {
         if (letters.length > MAX_LENGTH)
            throw new Exception("requested word length exceeds max(" + MAX_LENGTH + "): " + letters.length);

         word = 0;
         length = (byte)letters.length;
         
         packAll(letters);
    }
    
    public void push(int letter) {
        word = (word << BITS_PER_LETTER) | letter;
        ++length;
    }

    public long pop() {
        long letter = word & POP_MASK;
        shorten(1);
        return letter;
    }
    
    public void packAll(int [] letters) {
        for (int i = 0; i < letters.length; ++i)
            push(letters[i]);
    }
    
    public int[] unpackAll() {
        int[] letters = new int[length];

        int shift = WORD_SPACE-(length*BITS_PER_LETTER); //first shift, increment latter
        for (int i = length-1; i > -1; --i) {
            //left shift to left end to remove earlier letters, right shift to right end to remove latter
            letters[length-1-i] = (int)((word << shift) >>> (WORD_SPACE-BITS_PER_LETTER));

            shift += BITS_PER_LETTER;
        }

        return letters;
    }

    public void shortenByFourierCoefficient() {
        shorten(2); //1 real/imag pair
    }
    
    public void shorten(int amount) {
        length -= amount;
        word = word >>> amount*BITS_PER_LETTER;
    }

    @Override
    public int compareTo(BitWord o) {
        return Long.compare(word, o.getWord().longValue());
    }
    
    @Override
    public boolean equals(Object other) {
        if (other instanceof BitWord)
            return compareTo((BitWord)other) == 0;
        return false;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(word);
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
    public String toString(BitWordLong.PrintFormat format) {
        switch (format) {
            case RAW:
                return String.valueOf(Long.toString(word));
            case BINARY:
                return String.format("%"+WORD_SPACE+"s", Long.toBinaryString(word)).replace(' ', '0');
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

    public String toStringUnigram() {
        long[] letters = new long[length];
        int shift = WORD_SPACE-(length*BITS_PER_LETTER);
        for (int i = length-1; i > -1; --i) {
            letters[length-1-i] = (word << shift) >>> (WORD_SPACE-BITS_PER_LETTER);
            shift += BITS_PER_LETTER;
        }

        StringBuilder str = new StringBuilder();
        for (int i = 0; i < length; i++){
            str.append((char)('A'+letters[i]));
        }

        return str.toString();
    }

    public String toStringBigram() {
        long[] letters = new long[length];
        int shift = WORD_SPACE-(length*BITS_PER_LETTER);
        for (int i = length-1; i > -1; --i) {
            letters[length-1-i] = (word << shift) >>> (WORD_SPACE-BITS_PER_LETTER);
            shift += BITS_PER_LETTER;
        }

        long[] letters2 = new long[length];
        int shift2 = WORD_SPACE/2-(length*2);
        for (int i = length-1; i > -1; --i) {
            letters2[length-1-i] = (word << shift2) >>> (WORD_SPACE-BITS_PER_LETTER);
            shift2 += BITS_PER_LETTER;
        }

        StringBuilder str = new StringBuilder();
        for (int i = 0; i < length; i++){
            str.append((char)('A'+letters2[i]));
        }
        str.append("+");
        for (int i = 0; i < length; i++){
            str.append((char)('A'+letters[i]));
        }

        return str.toString();
    }

    public static void main(String [] args) throws Exception {
        int [] letters = {2,1,2,3,2,1,2,0,1,1,2,3,1,2,3,1};
        BitWordLong b = new BitWordLong();
        for (int i = 0; i < letters.length; i++){
            b.push(letters[i]);
            System.out.println(b.toString(BitWordLong.PrintFormat.BINARY));
        }

//        b.shorten(6);
//        System.out.println(b.toString(BitWordLong.PrintFormat.BINARY));

        b.word = (b.word << 32) | b.word;
        System.out.println(b.toString(BitWordLong.PrintFormat.BINARY));

        System.out.println();

        BitWordInt b2 = new BitWordInt();
        for (int i = 0; i < letters.length; i++){
            b2.push(letters[i]);
            System.out.println(b2.toString(BitWordInt.PrintFormat.BINARY));
        }

        BitWordLong b3 = new BitWordLong();
        System.out.println(b3.toString(BitWordLong.PrintFormat.BINARY));
        b3.word = (b2.getWord().longValue() << 32);
        System.out.println(b3.toString(BitWordLong.PrintFormat.BINARY));
        b3.word = b.word | Integer.toUnsignedLong(b2.getWord().intValue());
        System.out.println(b3.toString(BitWordLong.PrintFormat.BINARY));

        System.out.println();

        System.out.println(b.word);
        System.out.println(b2.getWord());
        System.out.println(b.compareTo(b2));

        System.out.println();

        System.out.println(Long.hashCode(10000L));
        System.out.println(Long.hashCode(10000000000001L));
        System.out.println(Long.hashCode(1316137241L));
    }
}
