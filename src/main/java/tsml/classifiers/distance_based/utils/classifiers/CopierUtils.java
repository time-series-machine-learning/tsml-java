package tsml.classifiers.distance_based.utils.classifiers;

import utilities.FileUtils;
import weka.core.SerializedObject;

import java.io.*;
import java.lang.annotation.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;
import java.util.zip.GZIPInputStream;

public class CopierUtils {

    public static final Predicate<Field> TRANSIENT = field -> Modifier.isTransient(field.getModifiers());

    /**
     * shallow copy an object, creating a new instance
     * @return
     * @throws Exception
     */
    public static Object shallowCopy(Object src) throws Exception {
        return shallowCopy(src, findFields(src.getClass()));
    }

    public static Object shallowCopy(Object src, Collection<Field> fields) throws Exception {
        // get the default constructor
        Constructor<?> noArgsConstructor = src.getClass().getDeclaredConstructor();
        // find out whether it's accessible from here (no matter if it's not)
        boolean origAccessible = noArgsConstructor.isAccessible();
        // force it to be accessible if not already
        noArgsConstructor.setAccessible(true);
        // use the constructor to build a default instance
        Object dest = noArgsConstructor.newInstance();
        // set the constructor's accessibility back to what it was
        noArgsConstructor.setAccessible(origAccessible);
        // copy over the fields from the current object to the new instance
        shallowCopyFrom(src, dest, fields);
        return dest;
    }

    /**
     * shallow copy fields from one object to another which already exists
     * @param src
     * @throws Exception
     */
    public static void shallowCopyFrom(Object src, Object dest) throws
            Exception {
        shallowCopyFrom(src, dest, findFields(src.getClass()));
    }

    public static void shallowCopyFrom(Object object, Object dest, Collection<Field> fields) throws
            Exception {
        copyFields(object, dest, false, fields);
    }

    // these are the same as the above, just deep versions

    public static Object deepCopy(Object src) throws Exception {
        return deepCopy(src, findFields(src.getClass()));
    }

    public static Object deepCopy(Object src, Collection<Field> fields) throws Exception {
        try {
            // newInstance() may not work if there's no default constructor / constructor throws an exception
            Object dest = src.getClass().newInstance();
            deepCopyFrom(src, dest, fields);
            return dest;
        } catch(Exception e) {
            // so try deep copying using serialisation instead. This ignores the list of fields to copy and will copy all fields instead!
            return serialisedDeepCopy(src);
        }
    }

    public static void deepCopyFrom(Object src, Object dest) throws
            Exception {
        deepCopyFrom(src, dest, findFields(src.getClass()));
    }

    public static void deepCopyFrom(Object src, Object dest, Collection<Field> fields) throws
            Exception {
        copyFields(src, dest, true, fields);
    }

    public static Object serialisedDeepCopy(Object src) throws IOException, ClassNotFoundException {
        return deserialise(serialise(src));
    }

    // copy functions for copying values

    public static byte[] serialise(Object obj) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(obj);
        return baos.toByteArray();
    }

    public static Object deserialise(byte[] bytes) throws IOException, ClassNotFoundException {
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        ObjectInputStream ois = new ObjectInputStream(bais);
        return ois.readObject();
    }

    public static <A> A copy(A object, boolean deep) throws
            Exception {
        if(deep) {
            return deepCopyValue(object);
        } else {
            return object;
        }
    }

    public static <A> A deepCopyValue(A object) throws
            Exception {
        return (A) new SerializedObject(object).getObject();
    }

    // copy several fields across to another object

    public static void copyFields(Object src, Object dest) throws
            Exception {
        copyFields(src, dest, false);
    }

    public static void copyFields(Object src, Object dest, Collection<Field> fields) throws Exception {
        copyFields(src, dest, false, fields);
    }

    public static void copyFields(Object src, Object dest, boolean deep, Collection<Field> fields) throws
            Exception {
        for(Field field : fields) {
            try {
                copyFieldValue(src, field, dest, deep);
            } catch(NoSuchFieldException e) {

            }
        }
    }

    public static void copyFields(Object src, Object dest, boolean deep) throws Exception {
        copyFields(src, dest, deep, findFields(dest.getClass()));
    }

    /**
     * get a field from an object
     * @param object
     * @param fieldName
     * @return
     * @throws NoSuchFieldException
     */
    public static Field getField(Object object, String fieldName) throws NoSuchFieldException {
        return getField(object.getClass(), fieldName);
    }

    /**
     * get field from class.
     * @param clazz
     * @param fieldName
     * @return
     * @throws NoSuchFieldException
     */
    public static Field getField(Class<?> clazz, String fieldName) throws NoSuchFieldException {
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
    public static Predicate<Field> STATIC = field -> {
        int modifiers = field.getModifiers();
        return Modifier.isStatic(modifiers);
    };
    public static Predicate<Field> FINAL = field -> {
        int modifiers = field.getModifiers();
        return Modifier.isFinal(modifiers);
    };
    public static Predicate<Field> DISABLE_COPY = field -> field.getAnnotation(DisableCopy.class) == null;
    // all fields except static and "no copy" fields (i.e. annotated with disable copy)
    public static Predicate<Field> DEFAULT_FIELDS = STATIC.negate().and(DISABLE_COPY);

    /**
     * find all non static fields in a class
     * @param clazz
     * @return
     */
    public static Set<Field> findFields(Class<?> clazz) {
        return findFieldsExcept(clazz, DEFAULT_FIELDS);
    }

    /**
     * find all fields in a class.
     * @param clazz
     * @return
     */
    public static Set<Field> findAllFields(Class<?> clazz) {
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
    public static Set<Field> findFieldsExcept(Class<?> clazz, Predicate<? super Field> predicate) {
        Set<Field> fields = findAllFields(clazz);
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
    public static Object copyFieldValue(Object src, Field field, Object dest, boolean deep) throws Exception {
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
    public static Object copyFieldValue(Object src, String fieldName, Object dest, boolean deep) throws Exception {
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
    public static void setFieldValue(Object object, String fieldName, Object value)
            throws NoSuchFieldException, IllegalAccessException {
        Field field = getField(object, fieldName);
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        field.set(object, value);
        field.setAccessible(accessible);
    }

    public static Object getFieldValue(Object object, String fieldName) throws NoSuchFieldException, IllegalAccessException {
        Field field = getField(object, fieldName);
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        final Object value = field.get(object);
        field.setAccessible(accessible);
        return value;
    }

    /**
     * find the fields which are not recorded in serialisation (as these don't need copying)
     * @param obj
     * @return
     */
    public static Set<Field> findSerialisableFields(Object obj) {
        return findFieldsExcept(obj.getClass(), TRANSIENT.negate().and(DEFAULT_FIELDS));
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
