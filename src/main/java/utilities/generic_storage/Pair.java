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
 * @param <T1>
 * @param <T2>
 * Generic Tuple class.
 */
public class Pair <T1, T2>{
    public T1 var1;
    public T2 var2;
    public Pair(T1 t1, T2 t2){
        var1 = t1;
        var2 = t2;
    }
    
    @Override
    public String toString(){
        return var1 + " " + var2;
    }
    
    @Override
    public boolean equals(/*Pair<T1,T2>*/Object ot){

        if (!(ot instanceof Pair)) 
            return false;
        
        Pair<T1, T2> other = (Pair<T1,T2>) ot;
        return var1.equals(other.var1) && var2.equals(other.var2);
    }
}
