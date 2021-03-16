package tsml.classifiers.distance_based.utils.collections.params;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import tsml.classifiers.distance_based.utils.system.copy.CopierUtils;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Function;

import static tsml.classifiers.distance_based.utils.strings.StrUtils.toOptionValue;

public class ParamSet implements Map<String, Object>, ParamHandler {
    
    private final Map<String, Object> map = new LinkedHashMap<>();

    public ParamSet(Map<String, Object> other) {
        putAll(other);
    }
    
    public ParamSet() {}
    
    public ParamSet(String[] options) throws Exception {
        setOptions(options);
    }
    
    public ParamSet(List<String> options) throws Exception {
        setOptions(options);
    }
    
    public ParamSet(String key, Object value, List<ParamSet> subParamSets) {
        add(key, value, subParamSets);   
    }
    
    public ParamSet(String key, Object value, ParamSet subParamSet) {
        add(key, value, subParamSet);
    }
    
    public ParamSet(String key, Object value) {
        add(key, value);
    }

    @Override public void setOptions(final List<String> options) throws Exception {
        setOptions(options.toArray(new String[options.size()]));
    }

    @Override public void setOptions(final String[] options) throws Exception {
        for(int i = 0; i < options.length; i++) {
            String option = options[i];
            if(option.isEmpty()) {
                // skip this option as it's empty
                continue;
            }
            String flag = StrUtils.unflagify(option);
            // if the flag is an option (i.e. key value pair, not just a flag)
            if(StrUtils.isOption(option, options)) {
                // for example, "-d "DTW -w 5""
                // get the next value as the option value and split it into sub options
                // the example would be split into ["DTW", "-w", "5"]
                String[] subOptions = Utils.splitOptions(options[++i]);
                // the 0th element is the main option value
                // in the example this is "DTW"
                String optionValue = subOptions[0];
                subOptions[0] = "";
                // get the value from str form
                Object value = StrUtils.fromOptionValue(optionValue);
                // if the return value is not a string, there's further options to set and the value can handle options
                if(subOptions.length > 1 && value instanceof OptionHandler) {
                    // handle sub parameters
                    ParamSet paramSet = new ParamSet();
                    // subOptions contains only the parameters for the option value
                    // in the example this is ["-w", "5"]
                    // set these suboptions for the parameter value
                    paramSet.setOptions(subOptions);
                    // add the parameter to this paramSet with correspond value (example "DTW") and corresponding
                    // sub options / parameters for that value (example "-w 5")
                    put(flag, value, paramSet);
                } else {
                    // the parameter is raw, i.e. "-a 6" <-- 6 has no parameters, therefore is raw
                    put(flag, value);
                }
            } else {
                // assume all flags are represented using boolean values
                put(flag, true);
            }
        }
    }

    @Override public int size() {
        return map.size();
    }

    @Override public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override public boolean containsKey(final Object o) {
        return map.containsKey(o);
    }

    @Override public boolean containsValue(final Object o) {
        return map.containsValue(o);
    }

    @Override public Object get(final Object o) {
        // must copy value otherwise can be changed externally
        return CopierUtils.deepCopy(map.get(o));
    }
    
    public <A> A get(String key) {
        return (A) get((Object) key);
    }

    @Override public Object put(final String s, final Object o) {
        // copy value so external changes to the value are not propagated in the paramset
        return map.put(s, CopierUtils.deepCopy(o));
    }
    
    public Object put(final String key, final Object value, ParamSet paramSet) {
        paramSet.applyTo(value);
        return put(key, value);
    }
    
    public ParamSet add(final String key, final Object value) {
        return add(key, value, new ParamSet());
    }
    
    public ParamSet add(final String key, final Object value, final ParamSet subParamSet) {
        final Object before = put(key, value, subParamSet);
        if(before != null) {
            throw new IllegalArgumentException("already have parameter set under key: " + key);
        }
        return this;
    }
    
    public ParamSet add(final String key, final Object value, final Iterable<ParamSet> subParamSets) {
        for(ParamSet paramSet : subParamSets) {
            paramSet.applyTo(value);   
        }
        return add(key, value);
    }
    
    public void applyTo(Object value) {
        if(this.isEmpty()) {
            return;
        }
        if(value instanceof ParamHandler) {
            try {
                ((ParamHandler) value).setParams(this);
            } catch(Exception e) {
                throw new IllegalArgumentException(e);
            }
        } else if(value instanceof OptionHandler) {
            try {
                ((OptionHandler) value).setOptions(this.getOptions());
            } catch(Exception e) {
                throw new IllegalArgumentException(e);
            }
        } else {
            throw new IllegalArgumentException("{" + value.toString() + "} is not a ParamHandler or OptionHandler therefore "
                                                       + "cannot set the parameters " + this.toString());
        }
    }

    @Override public String toString() {
        return Utils.joinOptions(getOptions());
    }
    
    @Override
    public List<String> getOptionsList() {
        List<String> list = new ArrayList<>();
        for(Map.Entry<String, ?> entry : entrySet()) {
            String name = entry.getKey();
            Object value = entry.getValue();
            list.add(StrUtils.flagify(name));
            list.add(toOptionValue(value));
        }
        return list;
    }

    public String toJson() {
        /*
        supported formats:
        {
          "a": 1,
          "b": [
            2,
            3
          ],
          "c": [
            [
              4
            ],
            [
              5
            ]
          ],
          "d": [
            [
              6,
              {
                "da": 6.1,
                "db": [
                  6.2,
                  6.3
                ]
              }
            ],
            [
              7,
              {
                "dc": 6.4,
                "dd": [
                  6.5,
                  6.6
                ]
              }
            ]
          ]
        }

        essentially, a param could:
            - map to a single value (a)
            - map to multiple values (b)
            - map to multiple values each contained in a array (c)
            - map to multiple values, each in their own array with a second obj in the array corresponding to their paramset in json form (d)
         */

        return new GsonBuilder().create().toJson(toJsonValue());
    }

    protected Map<String, Object> toJsonValue() {
        final HashMap<String, Object> output = new HashMap<>();
        for(String name : keySet()) {
            Object value = get(name);
            final ParamSet params = ParamHandlerUtils.getParams(value);
            if(!params.isEmpty()) {
                value = Arrays.asList(value, params.toJsonValue());
            }
            output.put(name, value);
        }
        return output;
    }
    
    protected static ParamSet fromJsonValue(Map<String, Object> json) {
        final ParamSet paramSet = new ParamSet();
        for(String name : json.keySet()) {
            Object value = json.get(name);
            if(value instanceof List<?>) {
                // list of 2 items: the value then the sub params
                final List<?> parts = (List<?>) value;
                value = parts.get(0);
                final ParamSet params = (ParamSet) parts.get(1);
                params.applyTo(value);
            }
            paramSet.put(name, value);
        }
        return paramSet;
    }

    public static ParamSet fromJson(String json) {
        final HashMap<?, ?> hashMap = new Gson().fromJson(json, HashMap.class);
        final ParamSet paramSet = new ParamSet();
        for(Map.Entry<?, ?> entry : hashMap.entrySet()) {
            final Object key = entry.getKey();
            final String keyStr;
            if(key instanceof String) {
                keyStr = (String) key;
            } else {
                throw new IllegalStateException("expected string type for key: " + key);
            }
            Object value = entry.getValue();
            ParamSet subParamSet = new ParamSet();
            if(value instanceof List<?>) {
                final List<?> parts = ((List<?>) value);
                value = parts.get(0);
                subParamSet = ParamSet.fromJsonValue((Map<String, Object>) parts.get(1));
            }
            paramSet.put(keyStr, value, subParamSet);
        }
        return paramSet;
    }
    
    @Override public Object remove(final Object o) {
        return map.remove(o);
    }

    @Override public void putAll(final Map<? extends String, ?> map) {
        this.map.putAll(map);
    }

    @Override public void clear() {
        map.clear();
    }

    @Override public Set<String> keySet() {
        return map.keySet();
    }

    @Override public Collection<Object> values() {
        return map.values();
    }

    @Override public Set<Entry<String, Object>> entrySet() {
        return map.entrySet();
    }

    @Override public boolean equals(final Object o) {
        return map.equals(o);
    }

    @Override public int hashCode() {
        return map.hashCode();
    }

    public <A> A get(String key, A defaultValue) {
        return (A) getOrDefault(key, defaultValue);
    }
    
    @Override public Object getOrDefault(final Object o, final Object defaultValue) {
        Object value = map.getOrDefault(o, defaultValue);
        if(value instanceof String) {
            value = StrUtils.parse((String) value, defaultValue.getClass());
        }
        return value;
    }

    @Override public void forEach(final BiConsumer<? super String, ? super Object> biConsumer) {
        map.forEach(biConsumer);
    }

    @Override public void replaceAll(final BiFunction<? super String, ? super Object, ?> biFunction) {
        map.replaceAll(biFunction);
    }

    @Override public Object putIfAbsent(final String s, final Object o) {
        return map.putIfAbsent(s, o);
    }

    @Override public boolean remove(final Object o, final Object o1) {
        return map.remove(o, o1);
    }

    @Override public boolean replace(final String s, final Object o, final Object v1) {
        return map.replace(s, o, v1);
    }

    @Override public Object replace(final String s, final Object o) {
        return map.replace(s, o);
    }

    @Override public Object computeIfAbsent(final String s, final Function<? super String, ?> function) {
        return map.computeIfAbsent(s, function);
    }

    @Override public Object computeIfPresent(final String s,
            final BiFunction<? super String, ? super Object, ?> biFunction) {
        return map.computeIfPresent(s, biFunction);
    }

    @Override public Object compute(final String s,
            final BiFunction<? super String, ? super Object, ?> biFunction) {
        return map.compute(s, biFunction);
    }

    @Override public Object merge(final String s, final Object o,
            final BiFunction<? super Object, ? super Object, ?> biFunction) {
        return map.merge(s, o, biFunction);
    }
}
