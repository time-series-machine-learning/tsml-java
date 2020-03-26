package tsml.classifiers.distance_based.utils.classifier_mixins;

import java.lang.reflect.Constructor;
import weka.core.SerializedObject;

import java.io.Serializable;
import java.lang.annotation.*;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.function.Predicate;

/**
 * Purpose: shallow and deep copy various fields from object to object using reflection. You can filter the fields to
 * ignore final fields / transient fields, etc, it's all flexible. Classes can implement this interface to provide
 * copy functionality but these functions can also be called statically and work in the same way. The benefit of the
 * former is being able to override the copy functions using inheritance, although that might not be in high demand.
 * Either way it works.
 *
 * Contributors: goastler
 */
public interface Copy extends Serializable {

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
        Constructor<? extends Copy> noArgsConstructor = getClass().getDeclaredConstructor();
        // find out whether it's accessible from here (no matter if it's not)
        boolean origAccessible = noArgsConstructor.isAccessible();
        // force it to be accessible if not already
        noArgsConstructor.setAccessible(true);
        // use the constructor to build a default instance
        Copy copy = noArgsConstructor.newInstance();
        // set the constructor's accessibility back to what it was
        noArgsConstructor.setAccessible(origAccessible);
        // copy over the fields from the current object to the new instance
        copy.shallowCopyFrom(this, fields);
        return copy;
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
        Copy copy = getClass().newInstance();
        copy.deepCopyFrom(this, fields);
        return copy;
    }

    default void deepCopyFrom(Object object) throws
                                             Exception {
        deepCopyFrom(object, findFields(this.getClass()));
    }

    default void deepCopyFrom(Object object, Collection<Field> fields) throws
                                             Exception {
        copyFields(object, this, true, fields);
    }

    // copy functions for copying values

    static <A> A copy(A object, boolean deep) throws
                                                    Exception {
        if(deep) {
            return deepCopy(object);
        } else {
            return object;
        }
    }

    static <A> A deepCopy(A object) throws
                                          Exception {
        return (A) new SerializedObject(object).getObject();
    }

    // copy several fields across to another object

    static void copyFields(Object src, Object dest) throws
                                                   Exception {
        copyFields(src, dest, false);
    }

    static void copyFields(Object src, Object dest, Collection<Field> fields) throws Exception {
        copyFields(src, dest, false, fields);
    }

    static void copyFields(Object src, Object dest, boolean deep, Collection<Field> fields) throws
                                                    Exception {
        for(Field field : fields) {
            try {
                copyFieldValue(src, field, dest, deep);
            } catch(NoSuchFieldException e) {

            }
        }
    }

    static void copyFields(Object src, Object dest, boolean deep) throws Exception {
        copyFields(src, dest, deep, findFields(dest.getClass(), DEFAULT_FIELDS));
    }

    /**
     * get a field from an object
     * @param object
     * @param fieldName
     * @return
     * @throws NoSuchFieldException
     */
    static Field getField(Object object, String fieldName) throws NoSuchFieldException {
        return getField(object.getClass(), fieldName);
    }

    /**
     * get field from class.
     * @param clazz
     * @param fieldName
     * @return
     * @throws NoSuchFieldException
     */
    static Field getField(Class<?> clazz, String fieldName) throws NoSuchFieldException {
        Field field = null;
        NoSuchFieldException ex = null;
        while(clazz != null && field == null) {
            try {
                field = clazz.getDeclaredField(fieldName);
            } catch(NoSuchFieldException e) {
                ex = e;
            }
            clazz = clazz.getSuperclass();
        }
        if(field == null) {
            throw ex;
        }
        return field;
    }

    // some default predicates for field types, these are used by default to avoid copying final / static fields and
    // avoid fields annotated with @DisableCopy
    Predicate<Field> FINAL_OR_STATIC = field -> {
        int modifiers = field.getModifiers();
        return Modifier.isFinal(modifiers) || Modifier.isStatic(modifiers);
    };
    Predicate<Field> COPY = field -> field.getAnnotation(DisableCopy.class) == null;
    Predicate<Field> DEFAULT_FIELDS = FINAL_OR_STATIC.negate().and(COPY);

    /**
     * find all fields in a class
     * @param clazz
     * @return
     */
    static Set<Field> findFields(Class<?> clazz) {
        Set<Field> fields = new HashSet<>();
        do {
            Collections.addAll(fields, clazz.getDeclaredFields());
            clazz = clazz.getSuperclass();
        } while (clazz != null);
        return fields;
    }

    /**
     * find fields in a class meeting given criteria
     * @param clazz
     * @param predicate
     * @return
     */
    static Set<Field> findFields(Class<?> clazz, Predicate<? super Field> predicate) {
        Set<Field> fields = findFields(clazz);
        fields.removeIf(predicate.negate());
        return fields;
    }

    /**
     * copy a field value from source to destination, ignoring accessibility protocol
     * @param src
     * @param field
     * @param dest
     * @param deep
     * @return
     * @throws Exception
     */
    static Object copyFieldValue(Object src, Field field, Object dest, boolean deep) throws Exception {
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        Object srcValue = field.get(src);
        Object destValue = copy(srcValue, deep);
        field.set(dest, destValue);
        field.setAccessible(accessible);
        return dest;
    }

    /**
     * copy field by name
     * @param src
     * @param fieldName
     * @param dest
     * @param deep
     * @return
     * @throws Exception
     */
    static Object copyFieldValue(Object src, String fieldName, Object dest, boolean deep) throws Exception {
        return copyFieldValue(src, getField(src, fieldName), dest, deep);
    }

    /**
     * set field value ignoring accessibility protocol
     * @param object
     * @param fieldName
     * @param value
     * @return
     * @throws NoSuchFieldException
     * @throws IllegalAccessException
     */
    static Object setFieldValue(Object object, String fieldName, Object value)
        throws NoSuchFieldException, IllegalAccessException {
        Field field = getField(object, fieldName);
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        field.set(object, value);
        field.setAccessible(accessible);
        return object;
    }

    /**
     * disable copy annotation. Put this above any field you *DO NOT* want to be copied. That field will be ignored
     * by default during copying, however this can be overriden.
     */
    @Retention(RetentionPolicy.RUNTIME) // accessible at runtime
    @Documented
    @Target(ElementType.FIELD) // only apply to fields
    @interface DisableCopy {
        String value() default "";
    }
}
