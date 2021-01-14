/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 

package statistics.transformations;
/*
 * Created on Jan 31, 2006
 *
 */

/**
 * @author ajb
 *
 */
public class ArrayPair implements Comparable {
    public double predicted;
    public double residual;
    public int compareTo(Object c) {
        if(this.predicted>((ArrayPair)c).predicted)
                return 1;
        else if(this.predicted<((ArrayPair)c).predicted)
                return -1;
        return 0;
    }

    public static void main(String[] args) {
    }
}
