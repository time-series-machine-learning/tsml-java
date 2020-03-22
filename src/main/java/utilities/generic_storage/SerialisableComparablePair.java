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

import java.io.Serializable;
import java.util.Objects;

public class SerialisableComparablePair<T1 extends Comparable<T1>, T2 extends Comparable<T2>>
    implements Comparable<SerialisableComparablePair<T1, T2>>, Serializable{

    public final T1 var1;
    public final T2 var2;

    protected static final long serialVersionUID = 389546738L;

    public SerialisableComparablePair(T1 t1, T2 t2){
        var1 = t1;
        var2 = t2;
    }
    
    @Override
    public String toString(){
        return var1 + " " + var2;
    }

    @Override
    public int compareTo(SerialisableComparablePair<T1, T2> other) {
        int c1 = var1.compareTo(other.var1);
        if (c1 != 0)
            return c1;
        else 
            return var2.compareTo(other.var2);
    }
    
    @Override
    public boolean equals(Object other) {
        if (other instanceof SerialisableComparablePair<?,?>)
            return var1.equals(((SerialisableComparablePair<?,?>)other).var1)
                    && var2.equals(((SerialisableComparablePair<?,?>)other).var2) ;
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(var1, var2);
    }
}
