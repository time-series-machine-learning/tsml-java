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
package timeseriesweka.classifiers;

/**
 * Interface that allows the user to impose a memory contract of a classifier that
    implements this interface

    known classifiers: None
 * @author pfm15hbu
 */
public interface MemoryContractable {
    enum DataUnit {BYTES, MEGABYTE, GIGABYTE}

    default void setSixGigabyteLimit(){ setMemoryLimit(DataUnit.GIGABYTE, 6); }

    default void setGigabyteLimit(int t){ setMemoryLimit(DataUnit.GIGABYTE, t); }

    default void setMegabyteLimit(int t){ setMemoryLimit(DataUnit.MEGABYTE, t); }

    //set any value in bytes you like.
    default void setMemoryLimit(long bytes){ setMemoryLimit(DataUnit.BYTES, bytes); }

    //pass in an value from the DataUnit enum and the amount of said values.
    void setMemoryLimit(DataUnit unit, long amount);
}
