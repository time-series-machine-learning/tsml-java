package tsml.classifiers.distance_based.utils.system.copy;

import java.io.Serializable;

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
    default <A> A shallowCopy() {
        return (A) CopierUtils.shallowCopy(this);
    }

    default void shallowCopyTo(Object dest) {
        CopierUtils.shallowCopy(this, dest);
    }

    default void shallowCopyTo(Object dest, Iterable<String> fields) {
        CopierUtils.shallowCopy(this, dest, fields);
    }

    default void shallowCopyTo(Object dest, String... fields) {
        CopierUtils.shallowCopy(this, dest, fields);
    }

    default void shallowCopyFrom(Object src) {
        CopierUtils.shallowCopy(src, this);
    }

    default void shallowCopyFrom(Object src, Iterable<String> fields) {
        CopierUtils.shallowCopy(src, this, fields);
    }

    default void shallowCopyFrom(Object src, String... fields) {
        CopierUtils.shallowCopy(src, this, fields);
    }

    default <A> A deepCopy() {
        return (A) CopierUtils.deepCopy(this);
    }

    default void deepCopyFrom(Object src) {
        CopierUtils.deepCopy(src, this);
    }

    default void deepCopyFrom(Object src, Iterable<String> fields) {
        CopierUtils.deepCopy(src, this, fields);
    }

    default void deepCopyFrom(Object src, String... fields) {
        CopierUtils.deepCopy(src, this, fields);
    }

    default void deepCopyTo(Object dest) {
        CopierUtils.deepCopy(this, dest);
    }

    default void deepCopyTo(Object dest, Iterable<String> fields) {
        CopierUtils.deepCopy(this, dest, fields);
    }

    default void deepCopyTo(Object dest, String... fields) {
        CopierUtils.deepCopy(this, dest, fields);
    }

}
