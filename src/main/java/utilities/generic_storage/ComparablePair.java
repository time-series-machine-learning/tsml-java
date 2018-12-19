package utilities.generic_storage;

import java.util.Objects;

public class ComparablePair <T1 extends Comparable<T1>, T2 extends Comparable<T2> > 
implements Comparable<ComparablePair<T1, T2> >{
    public final T1 var1;
    public final T2 var2;
    public ComparablePair(T1 t1, T2 t2){
        var1 = t1;
        var2 = t2;
    }
    
    @Override
    public String toString(){
        return var1 + " " + var2;
    }

    @Override
    public int compareTo(ComparablePair<T1, T2> other) {
        int c1 = var1.compareTo(other.var1);
        if (c1 != 0)
            return c1;
        else 
            return var2.compareTo(other.var2);
    }
    
    @Override
    public boolean equals(Object other) {
        if (other instanceof ComparablePair<?,?>)
            return var1.equals(((ComparablePair<?,?>)other).var1) 
                    && var2.equals(((ComparablePair<?,?>)other).var2) ;
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(var1, var2);
    }
}
