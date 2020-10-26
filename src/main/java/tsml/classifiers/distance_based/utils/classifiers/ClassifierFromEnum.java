package tsml.classifiers.distance_based.utils.classifiers;

import tsml.classifiers.EnhancedAbstractClassifier;

import java.util.Objects;

public interface ClassifierFromEnum<A extends EnhancedAbstractClassifier> extends Configurer<A>, Builder<A> {

    @Override <B extends A> B configure(B classifier);

    @Override default A build() {
        // assume that the enum is inside the classifier. Therefore, we're currently at an entry in the enum. We need to go up two classes (1 to the enum, 1 to the containing class, i.e. the classifier) then construct an instance of that class
        Class<?> parentClass = getClass().getSuperclass();
        if(Objects.isNull(parentClass)) {
            throw new IllegalStateException("no parent class found");
        }
        Class<?> grandParentClass = parentClass.getSuperclass();
        if(Objects.isNull(grandParentClass)) {
            throw new IllegalStateException("no grand parent class found");
        }
        try {
            A instance = (A) grandParentClass.newInstance();
            configure(instance);
            return instance;
        } catch(InstantiationException | IllegalAccessException e) {
            IllegalStateException ise =
                    new IllegalStateException("cannot instantiate grand parent class");
            ise.addSuppressed(e);
            throw ise;
        }
    }
}
