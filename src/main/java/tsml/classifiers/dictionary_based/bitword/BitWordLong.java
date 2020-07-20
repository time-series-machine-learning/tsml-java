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
public class BitWordLong implements Comparable<BitWordLong>, Serializable {

    protected static final long serialVersionUID = 1L;

    protected long word;

    //ALPHABETSIZE CURRENTLY ASSUMED TO BE 4, THEREFORE 2 BITS PER LETTER, AND MAX WORD LENGTH 32
    public static final int WORD_SPACE = 64; //length of long
    public static final int BITS_PER_LETTER = 2;
    public static final int MAX_LENGTH = 32; //WORD_SPACE/BITS_PER_LETTER;

    public BitWordLong() {
        word = 0L;
    }

    public BitWordLong(BitWordLong bw) {
        this.word = bw.word;
    }

    public BitWordLong(BitWordLong bw1, BitWordLong bw2){
        word = (bw1.word << 32) | bw2.word;
    }
    
    public void push(int letter) {
        word = (word << BITS_PER_LETTER) | letter;
    }
    
    public void shorten(int amount) {
        word = word >>> amount*BITS_PER_LETTER;
    }

    @Override
    public int compareTo(BitWordLong o) {
        return Long.compare(word, o.word);
    }

    public long getWord() { return word; }

    public void setWord(long l) { word = l; }
    
    @Override
    public boolean equals(Object other) {
        if (other instanceof BitWordLong)
            return compareTo((BitWordLong)other) == 0;
        return false;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(word);
    }
    
    @Override
    public String toString() {
        return String.format("%"+WORD_SPACE+"s", Long.toBinaryString((Long)word)).replace(' ', '0');
    }

    public String toStringUnigram(int length) {
        long[] letters = new long[length];
        int shift = 64-(length*2);
        for (int i = length-1; i > -1; --i) {
            letters[length-1-i] = (word << shift) >>> 62;
            shift += 2;
        }

        StringBuilder str = new StringBuilder();
        for (int i = 0; i < length; i++){
            str.append((char)('A'+letters[i]));
        }

        return str.toString();
    }

    public String toStringBigram(int length) {
        long[] letters = new long[length];
        int shift = 64-(length*2);
        for (int i = length-1; i > -1; --i) {
            letters[length-1-i] = (word << shift) >>> 62;
            shift += 2;
        }

        long[] letters2 = new long[length];
        int shift2 = 32-(length*2);
        for (int i = length-1; i > -1; --i) {
            letters2[length-1-i] = (word << shift2) >>> 62;
            shift2 += 2;
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
        int [] letters = {2,1,2,3,0};
        BitWordLong b = new BitWordLong();
        for (int i = 0; i < letters.length; i++){
            b.push(letters[i]);
            System.out.println(b);
        }

        int [] letters2 = {0,3,2,1,3};
        BitWordLong b2 = new BitWordLong();
        for (int i = 0; i < letters2.length; i++){
            b2.push(letters2[i]);
        }
        BitWordLong b3 = new BitWordLong(b,b2);
        System.out.println(b3);

        System.out.println(b.toStringUnigram(letters.length));
        System.out.println(b2.toStringUnigram(letters2.length));
        System.out.println(b3.toStringBigram(letters.length));
    }
}
