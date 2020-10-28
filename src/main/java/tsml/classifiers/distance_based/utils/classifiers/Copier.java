package tsml.classifiers.distance_based.utils.classifiers;

import java.io.IOException;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

/**
 * Purpose: shallow and deep copy various fields from object to object using reflection. You can filter the fields to
 * ignore final fields / transient fields, etc, it's all flexible. Classes can implement this interface to provide
 * copy functionality but these functions can also be called statically and work in the same way. The benefit of the
 * former is being able to override the copy functions using inheritance, although that might not be in high demand.
 * Either way it works.
 *
 * Contributors: goastler
 */
public interface Copier extends Serializable {

    /**
     * shallow copy an object, creating a new instance
     * @return
     * @throws Exception
     */
    default Object shallowCopy()
            throws NoSuchMethodException, IllegalAccessException, InstantiationException, ClassNotFoundException,
                           InvocationTargetException, IOException, NoSuchFieldException {
        return CopierUtils.shallowCopy(this);
    }

    default void shallowCopyTo(Object dest)
            throws IllegalAccessException, InvocationTargetException, InstantiationException, ClassNotFoundException,
                           NoSuchMethodException, IOException, NoSuchFieldException {
        CopierUtils.shallowCopy(this, dest);
    }

    default void shallowCopyTo(Object dest, Iterable<String> fields)
            throws IllegalAccessException, InstantiationException, IOException, NoSuchMethodException,
                           InvocationTargetException, ClassNotFoundException, NoSuchFieldException {
        CopierUtils.shallowCopy(this, dest, fields);
    }

    default void shallowCopyTo(Object dest, String... fields)
            throws IllegalAccessException, InstantiationException, IOException, NoSuchMethodException,
                           InvocationTargetException, ClassNotFoundException, NoSuchFieldException {
        CopierUtils.shallowCopy(this, dest, fields);
    }

    default void shallowCopyFrom(Object src)
            throws IllegalAccessException, InvocationTargetException, InstantiationException, ClassNotFoundException,
                           NoSuchMethodException, IOException, NoSuchFieldException {
        CopierUtils.shallowCopy(src, this);
    }

    default void shallowCopyFrom(Object src, Iterable<String> fields)
            throws IllegalAccessException, InstantiationException, IOException, NoSuchMethodException,
                           InvocationTargetException, ClassNotFoundException, NoSuchFieldException {
        CopierUtils.shallowCopy(src, this, fields);
    }

    default void shallowCopyFrom(Object src, String... fields)
            throws IllegalAccessException, InstantiationException, IOException, NoSuchMethodException,
                           InvocationTargetException, ClassNotFoundException, NoSuchFieldException {
        CopierUtils.shallowCopy(src, this, fields);
    }

    default Object deepCopy()
            throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchFieldException,
                           InstantiationException, IllegalAccessException {
        return CopierUtils.deepCopy(this);
    }

    default void deepCopyFrom(Object src)
            throws IllegalAccessException, InvocationTargetException, InstantiationException, ClassNotFoundException,
                           NoSuchMethodException, IOException, NoSuchFieldException {
        CopierUtils.deepCopy(src, this);
    }

    default void deepCopyFrom(Object src, Iterable<String> fields)
            throws IllegalAccessException, InstantiationException, IOException, NoSuchMethodException,
                           InvocationTargetException, ClassNotFoundException, NoSuchFieldException {
        CopierUtils.deepCopy(src, this, fields);
    }

    default void deepCopyFrom(Object src, String... fields)
            throws IllegalAccessException, InstantiationException, IOException, NoSuchMethodException,
                           InvocationTargetException, ClassNotFoundException, NoSuchFieldException {
        CopierUtils.deepCopy(src, this, fields);
    }

    default void deepCopyTo(Object dest)
            throws IllegalAccessException, InvocationTargetException, InstantiationException, ClassNotFoundException,
                           NoSuchMethodException, IOException, NoSuchFieldException {
        CopierUtils.deepCopy(this, dest);
    }

    default void deepCopyTo(Object dest, Iterable<String> fields)
            throws IllegalAccessException, InstantiationException, IOException, NoSuchMethodException,
                           InvocationTargetException, ClassNotFoundException, NoSuchFieldException {
        CopierUtils.deepCopy(this, dest, fields);
    }

    default void deepCopyTo(Object dest, String... fields)
            throws IllegalAccessException, InvocationTargetException, InstantiationException, ClassNotFoundException,
                           NoSuchMethodException, IOException, NoSuchFieldException {
        CopierUtils.deepCopy(this, dest, fields);
    }

}
