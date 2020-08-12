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

/**
 * Interface for BitWord classes
 *
 * @author Matthew Middlehurst
 */
public interface BitWord extends Comparable<BitWord>, Serializable {
    Number getWord();
    byte getLength();

    void setWord(Number word);

    void push(int letter);
    void shorten(int amount);
}
