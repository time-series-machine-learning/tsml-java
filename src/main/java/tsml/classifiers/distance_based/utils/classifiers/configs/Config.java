package tsml.classifiers.distance_based.utils.classifiers.configs;

import java.util.Objects;

/**
 * A config is targeted at a specific class, e.g. Proximity Tree. Then, given an object which is Proximity Tree or more specialised (extends PT) this config can configure it. Similarly, the default builder in this config may create instances which are Proximity Tree or more specialised (extends PT). The configurer may configure Proximity Tree or any class super to PT, e.g. configuring something in EAC say. This is fine because we know we always have at least a PT and at most something that extends PT, so a configurer which acts on anything in the inheritance hierarchy above PT is fine.
 * @param <A>
 */
public class Config<A> implements Builder<A>, Configurer<A> {
    
    public Config(final String name, final String description, final Builder<? extends A> template,
            final Configurer<? super A> configurer) {
        this.name = Objects.requireNonNull(name);
        this.description = description == null ? "" : description;
        this.template = Objects.requireNonNull(template);
        this.configurer = Objects.requireNonNull(configurer);
        if(name.length() == 0) {
            throw new IllegalArgumentException("empty name not allowed");
        }
    }

    private final String name;
    private final String description;
    /**
     * Builds a default instance which the configuration can then be applied to. I.e. take Proximity Forest. This method would return a new PF instance with r=5, say. During the build() or configure() method, the r=5 is changed to r=10, say, as defined by this configuration setting. Thus this method just returns a fresh instance which has default parameters, etc, set and this configuration has not been applied.
     * @return
     */
    private final Builder<? extends A> template;
    private final Configurer<? super A> configurer;
    
    public void configure(A obj) {
        configurer.configure(obj);
    }

    public final A build() {
        final A inst = template.build();
        configure(inst);
        return inst;
    }

    public String name() {
        return name;
    }

    public String description() {
        return description;
    }

    @Override public String toString() {
        return name;
    }
}
