package tsml.classifiers.distance_based.utils.classifiers;

import weka.classifiers.Classifier;

/**
 * Purpose: factory to produce classifier builders. Anything added to this factory is checked to ensure there is a
 * corresponding final field in the factory by the same name as the builder. E.g.
 * public static class Factory {
 *     public static final MY_CLSF = add(<classifier builder with the name MY_CLSF);
 * }
 * <p>
 * Contributors: goastler
 */
public class CompileTimeClassifierBuilderFactory<B extends Classifier> extends ClassifierBuilderFactory<B> {

//    public ClassifierBuilder<? extends B> add(ClassifierBuilder<? extends B> builder) {
//        try {
//            // must have field matching builder's name
//            Field field = this.getClass().getDeclaredField(builder.getName());
//            if(!Modifier.isFinal(field.getModifiers())) {
//                throw new IllegalStateException(builder.getName() + "field not final");
//            }
//        } catch(NoSuchFieldException e) {
//            throw new IllegalStateException(e);
//        }
//        super.add(builder);
//        return builder;
//    }

}
