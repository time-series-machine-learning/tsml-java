package tsml.classifiers.distance_based.utils.classifier_building;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import tsml.classifiers.distance_based.elastic_ensemble.ElasticEnsemble;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory.ClassifierBuilder;

/**
 * Purpose: factory to produce classifier builders. Anything added to this factory is checked to ensure there is a
 * corresponding final field in the factory by the same name as the builder. E.g.
 * public static class Factory {
 *     public static final MY_CLSF = add(<classifier builder with the name MY_CLSF);
 * }
 * <p>
 * Contributors: goastler
 */
public class CompileTimeClassifierBuilderFactory extends ClassifierBuilderFactory {

    public ClassifierBuilder add(ClassifierBuilder builder) {
        try {
            // must have field matching builder's name
            Field field = this.getClass().getDeclaredField(builder.getName());
            if(!Modifier.isFinal(field.getModifiers())) {
                throw new IllegalStateException(builder.getName() + "field not final");
            }
        } catch(NoSuchFieldException e) {
            throw new IllegalStateException(e);
        }
        add(builder);
        return builder;
    }

}
