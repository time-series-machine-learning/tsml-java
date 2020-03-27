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

import java.util.Objects;

public class ComparableKeyPair<T1 extends Comparable<T1>, T2 extends Comparable<T2> >
    implements Comparable<ComparableKeyPair<T1, T2>>{

    public final T1 var1;
    public final T2 var2;

    public ComparableKeyPair(T1 t1, T2 t2){
        var1 = t1;
        var2 = t2;
    }
    
    @Override
    public String toString(){
        return var1 + " " + var2;
    }

    @Override
    public int compareTo(ComparableKeyPair<T1, T2> other) {
        return var1.compareTo(other.var1);
    }
    
    @Override
    public boolean equals(Object other) {
        if (other instanceof ComparableKeyPair<?,?>)
            return var1.equals(((ComparableKeyPair<?,?>)other).var1);
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(var1);
    }
}
