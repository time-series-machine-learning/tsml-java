package utilities;

import weka.core.SerializedObject;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public interface Copy extends Serializable {
    default @NotNull Object shallowCopy() throws Exception {
        Copy copy = getClass().newInstance();
        copy.shallowCopyFrom(this);
        return copy;
    }

    default void shallowCopyFrom(@NotNull Object object) throws
                                                         Exception {
        copyFields(object, this);
    }

    default @NotNull Object deepCopy() throws
                                       Exception {
        return deepCopy(this);
    }

    default void deepCopyFrom(@NotNull Object object) throws
                                                      Exception {
        shallowCopyFrom(deepCopy(object));
    }

    static Object deepCopy(Object object) throws
                                              Exception {
        return new SerializedObject(object).getObject();
    }

    static <A> A deepCopy(Object object, Class<? extends A> clazz) throws
                                                                         Exception {
        return clazz.cast(deepCopy(object));
    }

    static Object copy(Object object, boolean deep) throws
                                                        Exception {
        if(deep) {
            return deepCopy(object);
        } else {
            return object;
        }
    }

    static void copyFields(Object src, Object dest) throws
                                                   Exception {
        copyFields(src, dest, false);
    }

    static void copyFields(Object src, Object dest, boolean deep) throws
                                                    Exception {
        List<Field> srcFields = findAllFields(src.getClass());
        List<Field> destFields = findAllFields(dest.getClass());
        for(Field srcField : srcFields) {
            for(Field destField : destFields) {
                if(srcField.equals(destField)) {
                    int modifiers = destField.getModifiers();
                    if(!Modifier.isFinal(modifiers) && !Modifier.isStatic(modifiers)) { // don't overwrite final /
                        // static fields
                        setFieldValue(src, srcField, dest, destField, deep);
                    }
                }
            }
        }
    }

    static List<Field> findAllFields(Class<?> clazz) {
        List<Field> fields = new ArrayList<>();
        do {
            Collections.addAll(fields, clazz.getDeclaredFields());
            clazz = clazz.getSuperclass();
        } while (clazz != null);
        return fields;
    }

    static Object setFieldValue(Object src, Field srcField, Object dest, Field destField, boolean deep) throws Exception {
        boolean srcAccessible = srcField.isAccessible();
        boolean destAccessible = destField.isAccessible();
        srcField.setAccessible(true);
        destField.setAccessible(true);
        Object srcValue = srcField.get(src);
        Object destValue = copy(srcValue, deep);
        destField.set(dest, destValue);
        srcField.setAccessible(srcAccessible);
        destField.setAccessible(destAccessible);
        return dest;
    }

    static Object setFieldValue(Object object, String name, Object value) {
        Field declaredField = null;
        try {
            declaredField = object.getClass().getDeclaredField(name);
        } catch(NoSuchFieldException e) {
            throw new IllegalStateException(e);
        }
        boolean accessible = declaredField.isAccessible();
        declaredField.setAccessible(true);
        try {
            declaredField.set(object, value);
        } catch(IllegalAccessException e) {
            throw new IllegalStateException("this shouldn't happen");
        }
        declaredField.setAccessible(accessible);
        return object;
    }

}
