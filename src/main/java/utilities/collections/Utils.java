package utilities.collections;

import java.util.Collection;
import java.util.Iterator;

public class Utils {
    private Utils() {}

    public static <A> A get(Iterator<A> iterator, int index) {
        if(index < 0) {
            throw new ArrayIndexOutOfBoundsException();
        }
        A result = null;
        for(int i = 0; i < index; i++) {
            if(!iterator.hasNext()) {
                throw new ArrayIndexOutOfBoundsException();
            }
            result = iterator.next();
        }
        return result;
    }

    public static <A> A get(Iterable<A> iterable, int index) {
        return get(iterable.iterator(), index);
    }
}
