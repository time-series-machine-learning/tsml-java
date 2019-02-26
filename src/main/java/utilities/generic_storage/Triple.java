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
package utilities.generic_storage;

/**
 *
 * @author raj09hxu
 */
public class Triple <T1, T2, T3> extends Pair<T1, T2>{
    public final T3 var3;
    public Triple(T1 t1, T2 t2, T3 t3){
        super(t1,t2);
        var3 = t3;
    }
    
    @Override
    public String toString(){
        return super.toString() + " " + var3;
    }
}
