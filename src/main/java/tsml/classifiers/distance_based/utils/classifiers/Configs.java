package tsml.classifiers.distance_based.utils.classifiers;

import tsml.classifiers.distance_based.utils.collections.iteration.TransformIterator;

import java.util.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

/**
 * A set of configs are targeted at a specific type, say PT for example. Each config is mapped by name to create a factory which can produce fresh instances of that type, e.g. PT.
 * @param <A>
 */
public class Configs<A> implements Iterable<Config<A>> {
    
    private final Map<String, Config<A>> map = new TreeMap<>(); // using treemap to maintain ordering of names

    private void put(String name, Config<A> config) {
        name = name.toUpperCase();
        if(map.containsKey(name)) {
            throw new IllegalArgumentException("mapping for " + name + " already exists");
        }
        map.put(name, config);
    }
    
    public void add(String alias, String target) {
        final Config<A> config = get(target);
        put(alias, new Config<>(alias, config.description(), config, config));
    }
    
    public void add(String name, String description, Builder<? extends A> template, Configurer<? super A> configurer) {
        add(new Config<>(name, description, template, configurer));
    }
    
    public void add(String name, String description, String templateName, Configurer<? super A> configurer) {
        final Config<A> template = get(templateName);
        add(name, template.description() + ". " + description, template, new Configurer<A>() {
            @Override public void configure(final A obj) {
                // configure the template
                template.configure(obj);
                // then apply the specialisation config over the template
                configurer.configure(obj);
            }
        });
    }
    
    public void add(Config<A> config) {
        put(config.name(), config);
    }
    
    public Config<A> remove(String name) {
        return map.remove(name.toUpperCase());
    }
    
    public Config<A> get(String name) {
        final Config<A> config = map.get(name.toUpperCase());
        if(config == null) {
            throw new IllegalArgumentException(name + " not found");
        }
        return config;
    }
    
    public void configure(String name, A obj) {
        get(name).configure(obj);
    }
    
    public A build(String name) {
        return get(name).build();
    }
    
    public boolean contains(String name) {
        return map.containsKey(name.toUpperCase());
    }
    
    public Set<String> keySet() {
        return Collections.unmodifiableSet(map.keySet());
    }

    @Override public String toString() {
        StringBuilder sb = new StringBuilder();
        final ArrayList<String> names = new ArrayList<>(keySet());
        for(int i = 0; i < names.size(); i++) {
            final Config<A> config = get(names.get(i));
            sb.append(config);
            if(i < names.size() - 1) {
                sb.append(System.lineSeparator());
            }
        }
        return sb.toString();
    }
    
    public Configs<A> immutable() {
        final Configs<A> current = this;
        return new Configs<A>() {
            @Override public void add(final String name, final String description, final Builder<? extends A> template,
                    final Configurer<? super A> configurer) {
                throw new UnsupportedOperationException();
            }

            @Override public void add(final String name, final String description, final String templateName,
                    final Configurer<? super A> configurer) {
                throw new UnsupportedOperationException();
            }

            @Override public void add(final Config<A> config) {
                throw new UnsupportedOperationException();
            }

            @Override public Config<A> remove(final String name) {
                throw new UnsupportedOperationException();
            }

            @Override public Config<A> get(final String name) {
                return current.get(name);
            }

            @Override public A build(final String name) {
                return current.build(name);
            }

            @Override public Set<String> keySet() {
                return current.keySet();
            }

            @Override public String toString() {
                return current.toString();
            }

            @Override public Configs<A> immutable() {
                return this;
            }
        };
    }
    
    public Map<String, Builder<A>> toBuilderMap() {
        final HashMap<String, Builder<A>> map = new HashMap<>();
        for(Config<A> config : this) {
            map.put(config.name(), config);
        }
        return map;
    }

    @Override public Iterator<Config<A>> iterator() {
        return new TransformIterator<>(keySet().iterator(), this::get);
    }
}
