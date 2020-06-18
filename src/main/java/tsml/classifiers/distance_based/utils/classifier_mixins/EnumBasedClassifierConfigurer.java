package tsml.classifiers.distance_based.utils.classifier_mixins;

import tsml.classifiers.EnhancedAbstractClassifier;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

public interface EnumBasedClassifierConfigurer<A extends EnhancedAbstractClassifier> extends ClassifierConfigurer<A> {

    @Override default <B extends A> B applyConfigTo(final B classifier) {
        try {
            if(!Copy.getFieldValue(this, "name").equals("DEFAULT")) {
                final Class<?> enclosingClass = getClass().getEnclosingClass();
                final Object[] enumConstants = enclosingClass.getEnumConstants();
                for(Object constant : enumConstants) {
                    final Object value = Copy.getFieldValue(constant, "name");
                    if(value.equals("DEFAULT")) {
                        ((EnumBasedClassifierConfigurer<A>) constant).applyConfigTo(classifier);
                        break;
                    }
                }
            }
        } catch(NoSuchFieldException | IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        return classifier;
    }

    @Override default String getClassifierName() {
        // defaults to the name of the enum constant
        return name();
    }

    String name();
}
