package utilities;

import scala.annotation.meta.field;
import weka.core.SerializedObject;

import java.io.Serializable;
import java.lang.annotation.*;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.function.Predicate;

public interface Copy extends Serializable {

    default Object shallowCopy() throws Exception {
        return shallowCopy(findFields(this.getClass()));
    }

    default Object shallowCopy(Collection<Field> fields) throws Exception {
        Copy copy = getClass().newInstance();
        copy.shallowCopyFrom(this, fields);
        return copy;
    }

    default void shallowCopyFrom(Object object) throws
                                                Exception {
        shallowCopyFrom(object, findFields(this.getClass()));
    }

    default void shallowCopyFrom(Object object, Collection<Field> fields) throws
                                                         Exception {
        copyFields(object, this, false, fields);
    }

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

    // copy functions

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

    static Field getField(Object object, String fieldName) throws NoSuchFieldException {
        return getField(object.getClass(), fieldName);
    }

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

    Predicate<Field> FINAL_OR_STATIC = field -> {
        int modifiers = field.getModifiers();
        return Modifier.isFinal(modifiers) || Modifier.isStatic(modifiers);
    };
    Predicate<Field> COPY = field -> field.getAnnotation(DisableCopy.class) == null;
    Predicate<Field> DEFAULT_FIELDS = FINAL_OR_STATIC.negate().and(COPY);

    static Set<Field> findFields(Class<?> clazz) {
        Set<Field> fields = new HashSet<>();
        do {
            Collections.addAll(fields, clazz.getDeclaredFields());
            clazz = clazz.getSuperclass();
        } while (clazz != null);
        return fields;
    }

    static Set<Field> findFields(Class<?> clazz, Predicate<? super Field> predicate) {
        Set<Field> fields = findFields(clazz);
        fields.removeIf(predicate.negate());
        return fields;
    }

    static Object copyFieldValue(Object src, Field field, Object dest, boolean deep) throws Exception {
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        Object srcValue = field.get(src);
        Object destValue = copy(srcValue, deep);
        field.set(dest, destValue);
        field.setAccessible(accessible);
        return dest;
    }

    static Object copyFieldValue(Object src, String fieldName, Object dest, boolean deep) throws Exception {
        return copyFieldValue(src, getField(src, fieldName), dest, deep);
    }

    static Object setFieldValue(Object object, String fieldName, Object value)
        throws NoSuchFieldException, IllegalAccessException {
        Field field = getField(object, fieldName);
        boolean accessible = field.isAccessible();
        field.setAccessible(true);
        field.set(object, value);
        field.setAccessible(accessible);
        return object;
    }

    @Retention(RetentionPolicy.RUNTIME) // accessible at runtime
    @Documented
    @Target(ElementType.FIELD) // only apply to fields
    @interface DisableCopy {
        String value() default "";
    }
}
