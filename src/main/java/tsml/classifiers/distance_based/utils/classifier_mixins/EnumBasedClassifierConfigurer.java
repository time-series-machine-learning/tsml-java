package tsml.classifiers.distance_based.utils.classifier_mixins;

import tsml.classifiers.EnhancedAbstractClassifier;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

public interface EnumBasedClassifierConfigurer<A extends EnhancedAbstractClassifier> extends ClassifierConfigurer<A> {

    @Override default String getClassifierName() {
        // defaults to the name of the enum constant
        return name();
    }

    String name();
}
