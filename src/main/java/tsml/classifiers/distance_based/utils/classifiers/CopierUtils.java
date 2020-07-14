package tsml.classifiers.distance_based.utils.classifiers;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.Collection;

public class CopierUtils {

    /**
     * shallow copy an object, creating a new instance
     * @return
     * @throws Exception
     */
    public static Object shallowCopy(Object src) {
        return shallowCopy(findFields(src.getClass()));
    }

    public static Object shallowCopy(Object src, Collection<Field> fields) throws Exception {
        // get the default constructor
        Constructor<?> noArgsConstructor = src.getClass().getDeclaredConstructor();
        // find out whether it's accessible from here (no matter if it's not)
        boolean origAccessible = noArgsConstructor.isAccessible();
        // force it to be accessible if not already
        noArgsConstructor.setAccessible(true);
        // use the constructor to build a default instance
        Object obj = noArgsConstructor.newInstance();
        // set the constructor's accessibility back to what it was
        noArgsConstructor.setAccessible(origAccessible);
        // copy over the fields from the current object to the new instance
        shallowCopyFrom(src, fields, dest);
        obj.shallowCopyFrom(this, fields);
        return obj;
    }

}
