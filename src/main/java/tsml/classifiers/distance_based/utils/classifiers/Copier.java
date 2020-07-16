package tsml.classifiers.distance_based.utils.classifiers;

import java.lang.reflect.Constructor;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.*;

import static tsml.classifiers.distance_based.utils.classifiers.CopierUtils.copyFields;
import static tsml.classifiers.distance_based.utils.classifiers.CopierUtils.findFields;

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
    default Object shallowCopy() throws Exception {
        return shallowCopy(findFields(this.getClass()));
    }

    default Object shallowCopy(Collection<Field> fields) throws Exception {
        // get the default constructor
        Constructor<? extends Copier> noArgsConstructor = getClass().getDeclaredConstructor();
        // find out whether it's accessible from here (no matter if it's not)
        boolean origAccessible = noArgsConstructor.isAccessible();
        // force it to be accessible if not already
        noArgsConstructor.setAccessible(true);
        // use the constructor to build a default instance
        Copier copier = noArgsConstructor.newInstance();
        // set the constructor's accessibility back to what it was
        noArgsConstructor.setAccessible(origAccessible);
        // copy over the fields from the current object to the new instance
        copier.shallowCopyFrom(this, fields);
        return copier;
    }

    /**
     * shallow copy fields from one object to another which already exists
     * @param object
     * @throws Exception
     */
    default void shallowCopyFrom(Object object) throws
                                                Exception {
        shallowCopyFrom(object, findFields(this.getClass()));
    }

    default void shallowCopyFrom(Object object, Collection<Field> fields) throws
                                                         Exception {
        copyFields(object, this, false, fields);
    }

    // these are the same as the above, just deep versions

    default Object deepCopy() throws
                                       Exception {
        return deepCopy(findFields(this.getClass()));
    }

    default Object deepCopy(Collection<Field> fields) throws Exception {
        Copier copier = getClass().newInstance();
        copier.deepCopyFrom(this, fields);
        return copier;
    }

    default void deepCopyFrom(Object object) throws
                                             Exception {
        deepCopyFrom(object, findFields(this.getClass()));
    }

    default void deepCopyFrom(Object object, Collection<Field> fields) throws
                                             Exception {
        copyFields(object, this, true, fields);
    }

}
