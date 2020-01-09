package utilities;

import weka.core.SerializedObject;

import java.io.Serializable;
import java.lang.reflect.Field;
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
                    boolean srcFieldAccessible = srcField.isAccessible();
                    boolean destFieldAccessible = destField.isAccessible();
                    if(srcFieldAccessible && destFieldAccessible) {
                        Object value = srcField.get(src);
                        destField.set(dest, copy(value, deep));
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
}
