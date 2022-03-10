package tsml.classifiers.distance_based.utils.system.copy;

import java.io.*;
import java.lang.annotation.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import static java.util.Collections.addAll;

public class CopierUtils {

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

    // some default predicates for field types, these are used by default to avoid copying final / static fields and
    // avoid transient fields. Transient fields are usually instance specific ties to tmp resources, hence why they cannot be saved to disk. As they're instance specific they are left out of the copying procedure.
    public static final Predicate<Field> TRANSIENT = field -> Modifier.isTransient(field.getModifiers());
    // avoid static fields (as these are init'd / managed outside of the instance and therefore should not be included in copying
    public static Predicate<Field> STATIC = field -> Modifier.isStatic(field.getModifiers());
    public static Predicate<Field> FINAL = field -> Modifier.isFinal(field.getModifiers());
    // custom annotation to disable copy (I.e. mark a field with @DisableCopy) to remove it from the default field list when copying
    // avoid fields annotated with @DisableCopy
    public static Predicate<Field> DISABLE_COPY = field -> field.getAnnotation(DisableCopy.class) != null;

    public static void shallowCopy(Object src, Object dest) {
        CopierUtils.shallowCopy(src, dest, findDefaultShallowCopyFieldNames(src));
    }

    public static void deepCopy(Object src, Object dest) {
        deepCopy(src, dest, findDefaultDeepCopyFieldNames(src));
    }

    public static List<Field> findDefaultShallowCopyFields(Object src) {
        final List<Field> fields = findFields(src);
        // do not deep copy static, transient or disable_copy fields
        fields.removeIf(STATIC.or(TRANSIENT).or(DISABLE_COPY));
        return fields;
    }

    public static List<String> findDefaultShallowCopyFieldNames(Object src) {
        return findDefaultShallowCopyFields(src).stream().map(Field::getName).collect(Collectors.toList());
    }

    public static List<Field> findDefaultDeepCopyFields(Object src) {
        final List<Field> fields = findFields(src);
        // do not deep copy static, transient or disable_copy fields
        fields.removeIf(STATIC.or(TRANSIENT).or(DISABLE_COPY));
        return fields;
    }

    public static List<String> findDefaultDeepCopyFieldNames(Object src) {
        return findDefaultDeepCopyFields(src).stream().map(Field::getName).collect(Collectors.toList());
    }

    public static void shallowCopy(Object src, Object dest, Iterable<String> fields) {
        if(src == null || dest == null) return;
        copyFieldValues(src, dest, false, fields);
    }

    public static void deepCopy(Object src, Object dest, Iterable<String> fields) {
        if(src == null || dest == null) return;
        copyFieldValues(src, dest, true, fields);
    }

    public static void deepCopy(Object src, Object dest, String... fields) {
        if(src == null || dest == null) return;
        deepCopy(src, dest, Arrays.asList(fields));
    }

    public static void shallowCopy(Object src, Object dest, String... fields) {
        if(src == null || dest == null) return;
        shallowCopy(src, dest, Arrays.asList(fields));
    }

    /**
     * copy several fields across to another object
     */
    private static void copyFieldValues(Object src, Object dest, boolean deep, Iterable<String> fields) {
        for(String field : fields) {
            copyFieldValue(src, dest, deep, field);
        }
    }

    public static void deepCopyFieldValues(Object src, Object dest, Iterable<String> fields) {
        copyFieldValues(src, dest, true, fields);
    }

    public static void shallowCopyFieldValues(Object src, Object dest, Iterable<String> fields) {
        copyFieldValues(src, dest, false, fields);
    }

    public static void shallowCopyFieldValues(Object src, Object dest, String... fields) {
        shallowCopyFieldValues(src, dest, Arrays.asList(fields));
    }

    public static void deepCopyFieldValues(Object src, Object dest, String... fields) {
        deepCopyFieldValues(src, dest, Arrays.asList(fields));
    }

    /**
     * copy a field value from source to destination, ignoring accessibility protocol
     * @param src
     * @param fields
     * @param dest
     * @param deep
     * @return
     * @throws Exception
     */
    private static void copyFieldValue(Object src, Object dest, boolean deep, String... fields) {
        for(String field : fields) {
            Object value = getFieldValue(src, field);
            value = copy(value, deep);
            setFieldValue(dest, field, value);
        }
    }

    /**
     * set field value ignoring accessibility protocol
     * @param dest
     * @param field
     * @param value
     * @return
     * @throws NoSuchFieldException
     * @throws IllegalAccessException
     */
    public static void setFieldValue(Object dest, Field field, Object value) {
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        try {
            field.set(dest, value);
        } catch(IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        field.setAccessible(accessible);
    }

    public static void setFieldValue(Object dest, String name, Object value) {
        setFieldValue(dest, getField(dest, name), value);
    }

    public static Object getFieldValue(Object src, Field field) {
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        try {
            final Object value = field.get(src);
            field.setAccessible(accessible);
            return value;
        } catch(IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    public static Object getFieldValue(Object src, String name) {
        return getFieldValue(src, getField(src, name));
    }

    public static byte[] serialise(Object obj) {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(obj);
            return baos.toByteArray();
        } catch(IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static <A> A deserialise(byte[] bytes) {
        try {
            ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
            ObjectInputStream ois = new ObjectInputStream(bais);
            return (A) ois.readObject();
        } catch(IOException | ClassNotFoundException e) {
            throw new IllegalStateException(e);
        }
    }

    private static <A> A copy(A object, boolean deep) {
        if(deep) {
            return deepCopy(object);
        } else {
            return object;
        }
    }

    public static <A> A shallowCopy(A src) {
        if(src == null) return null;
        return shallowCopyViaDefaultConstructor(src);
    }

    public static <A> A newInstanceFromClassName(String className) {
        try {
            return newInstance(Class.forName(className));
        } catch(ClassCastException | ClassNotFoundException e) {
            throw new IllegalStateException(e);
        }
    }

    public static <A> A newInstance(A object) {
        if(object == null) {
            return null;
        }
        return newInstance(object.getClass());
    }

    public static <A> A newInstance(Class<?> clazz) {
        try {
            // get the default constructor
            final Constructor<?> noArgsConstructor = clazz.getDeclaredConstructor();
            // find out whether it's accessible from here (no matter if it's not)
            final boolean origAccessible = noArgsConstructor.isAccessible();
            // force it to be accessible if not already
            noArgsConstructor.setAccessible(true);
            // use the constructor to build a default instance
            final Object inst = noArgsConstructor.newInstance();
            // set the constructor's accessibility back to what it was
            noArgsConstructor.setAccessible(origAccessible);
            return (A) inst;
        } catch(ClassCastException | InstantiationException | InvocationTargetException | NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    private static <A> A shallowCopyViaDefaultConstructor(A src) {
        if(src == null) return null;
        A dest = newInstance(src);
        // copy over the fields from the current object to the new instance
        shallowCopy(src, dest);
        return dest;
    }

    public static <A> A deepCopy(A src) {
        if(src == null) return null;
        // quick check for boxed primitives / immutable objects, as these don't need copying
        if(src instanceof String
                || src instanceof Double
                   || src instanceof Integer
                   || src instanceof Long
                   || src instanceof Float
                   || src instanceof Byte
                   || src instanceof Boolean
                   || src instanceof Character
                   || src instanceof Short
        ) {
            return src;
        }
        // deep copy the source
        src = deserialise(serialise(src));
        try {
            // then attempt to invoke the default constructor to make a new instance and copy the deeply copied src into the new instance
            // this is necessary so a new instance is created and the default constructor is run, initialising any transient variables not copied during the serialisation copy
            return shallowCopyViaDefaultConstructor(src);
        } catch(IllegalStateException e) {
            // if there's no default constructor then use the deep copy already made. This means the default constructor will not be called and transient field may be left null. Users will have to account for this when using transient fields
            return src;
        }
    }

    public static List<Field> findFields(Object src) {
        return findFields(src.getClass());
    }

    public static List<Field> findFields(Class<?> clazz) {
        final Set<Field> fields = new HashSet<>();
        do {
            addAll(fields, clazz.getDeclaredFields());
            clazz = clazz.getSuperclass();
        } while(clazz != null);
        return new ArrayList<>(fields);
    }

    public static Field getField(Object obj, String name) {
        return getField(obj.getClass(), name);
    }

    public static Field getField(Class<?> clazz, String name) {
        Field field = null;
        IllegalStateException ex = null;
        while(clazz != null && field == null) {
            try {
                field = clazz.getDeclaredField(name);
            } catch(NoSuchFieldException e) {
                ex = new IllegalStateException(e);
            }
            clazz = clazz.getSuperclass();
        }
        if(field == null) {
            throw ex;
        }
        return field;
    }

}
