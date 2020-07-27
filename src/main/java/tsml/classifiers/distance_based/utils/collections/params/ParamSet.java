package tsml.classifiers.distance_based.utils.collections.params;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import weka.core.Utils;

/**
 * Purpose: store a mapping of parameter names to their corresponding values.
 *
 * Note, this parameter mapping stuff can be a little confusing so read this first!
 * We maintain a map of parameter names to a list of values. There are a list of values because some parameters may
 * be able to accept lists / arrays, therefore we have a 1 to many mapping. This list stores objects. This was chosen
 * over a generic implementation due to the complexity and reduction in versatility by making it generic. I.e. it's
 * very easy to handle a ParamSet with objects as the value, passing it to other classes which accept ParamSets. If
 * it were generic, the layer of complexity to pass around generic ParamSets and potential compatibility issues make
 * it not worth it. Further, this way we can store values in any which form (e.g. Strings from the command line).
 * These will be passed inside the implementation class appropriately, therefore it doesn't matter that we have
 * parameters in String or primitive type format - either work. That's also a win on memory, as storing all values as
 * Strings takes up a fair amount of memory / time to convert primitive to String and back.
 *
 * Another complexity of the ParamSet structure is the fact that the values for each parameter themselves could be
 * ParamSets. This is to support sub parameter sets, e.g. if a KNN has a distance measure, but that distance measure
 * also takes parameters. We need to be able to specify the distance measure value for the KNN *and* the parameters
 * for the distance measure all in one ParamSet. In this case we would have 1 ParamSet mapping "dm" (for distance
 * measure, say) to an Object of type DistanceMeasure (say, an instance of DTW). The DTW instance then houses its own
 * parameters in the same way. When the KNN receives a ParamSet through setParams, the distance measure is assigned
 * the corresponding parameter value and the parameter for the distance measure are passed on from the KNN.
 *
 * Contributors: goastler
 */
public class ParamSet implements ParamHandler, Serializable {

    public ParamSet() {

    }

    public ParamSet(String name, Object value, List<ParamSet> paramSets) {
        add(name, value, paramSets);
    }

    public ParamSet(String name, Object value, ParamSet paramSet) {
        this(name, value, Arrays.asList(paramSet));
    }

    public ParamSet(String name, Object value) {
        add(name, value);
    }

    public ParamSet(ParamSet paramSet) {
        addAll(paramSet);
    }

    @Override
    public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(o == null || getClass() != o.getClass()) {
            return false;
        }
        final ParamSet paramSet = (ParamSet) o;
        return paramMap.equals(paramSet.paramMap);
    }

    @Override
    public int hashCode() {
        return Objects.hash(paramMap);
    }

    private Map<String, List<Object>> paramMap = new HashMap<>();

    public int size() {
        int size = 0;
        for(Map.Entry<String, List<Object>> entry : paramMap.entrySet()) {
            List<Object> values = entry.getValue();
            for(Object value : values) {
                if(value instanceof ParamSet) {
                    size += ((ParamSet) value).size();
                } else {
                    size++;
                }
            }
        }
        return size;
    }

    public boolean isEmpty() {
        return paramMap.isEmpty();
    }

    /**
     * NOTE: the values associated with each parameter name may be strings. This is especially likely when the options come
     * from the command line. It is down to the client to handle the fact that the object returned by the get() method
     * may be a string instead of the object they are intending. E.g. say I have the ParamSet { "w" -> 5 }. 5 is an
     * integer so I can handle it normally. However, if this ParamSet had come off the cmdline then it would be { "w"
     * -> "5" } as there is no context to what the value of "w" (i.e. "5") should be (is it a long / double / int /
     * String?). Consequently, the value you expect back from this function may be as a string rather than the type
     * you're expecting. You must do a "x instanceof String" check and parse from a string as necessary / required.
     *
     * @param name
     * @return
     */
    public List<Object> get(String name) {
        return paramMap.get(name);
    }

    public Object getSingle(String name) {
        final List<Object> list = get(name);
        if(list == null) {
            return null;
        }
        Assert.assertEquals(1, list.size());
        return list.get(0);
    }

    public boolean contains(String name) {
        return paramMap.get(name) != null;
    }

    public ParamSet add(String name, Object value) {
        paramMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
        return this;
    }

    public ParamSet add(String name, Object value, ParamSet params) {
        add(name, value, Collections.singletonList(params));
        return this;
    }

    /**
     * add a parameter name mapping to a parameter value. The value is a parameter handler which accepts the ParamSet
     * . The ParamSet is applied to the value during this function. An example would be ("DTW", new DTW(), dtwParams)
     * . The DTW instance accepts the dtwParams object which sets, say, the warping window. Back here, the "DTW"
     * string is mapped to the DTW instance. All parameters / configuration are therefore housed in a map using
     * containment.
     *  If multiple ParamSets are specified, each one is applied to the value in turn
     * @param name
     * @param value
     * @param params
     * @return
     */
    public ParamSet add(String name, Object value, List<ParamSet> params) {
        setParams(value, params);
        return add(name, value);
    }

    public static void setParams(Object value, ParamSet param) {
        setParams(value, Collections.singletonList(param));
    }

    public static void setParams(Object value, List<ParamSet> params) {
        if(!params.isEmpty()) {
            if(value instanceof ParamHandler) {
                for(ParamSet param : params) {
                    try {
                        ((ParamHandler) value).setParams(param);
                    } catch(Exception e) {
                        throw new IllegalArgumentException(e);
                    }
                }
            } else {
                throw new IllegalArgumentException("{" + value.toString() + "} is not a ParamHandler therefore "
                    + "cannot set the parameters {" + params.toString() + "}");
            }
        }
    }

    public ParamSet addAll(ParamSet paramSet) {
        for(Entry<String, List<Object>> entry : paramSet.paramMap.entrySet()) {
            String key = entry.getKey();
            List<Object> list = entry.getValue();
            for(Object value : list) {
                add(key, value);
            }
        }
        return this;
    }

    public ParamSet clear() {
        paramMap.clear();
        return this;
    }

    // todo make paramSet / paramSpace compatible with flags (urgh flags are naff) (weka uses flags for boolean
    //  values, these are very difficult to parse, especially when it comes to neg numbers. We need to implement
    //  flags, but also check whether the value after the flag is an option or not. If it is, that flag is for a
    //  bespoke type, say int, and if not the flag's presence means the boolean is on. The tough part comes with the
    //  flags absence, as this corresponds to either a) boolean is false or b) the user forgot to specify that
    //  argument, as they don't currently have to specify every parameter when setting params. I think the only way
    //  around this is to either force booleans to be parameter based (followed by t/f) or force that everytime
    //  params are set boolean values need to be specified. *or* we have a boolean which controls whether the
    //  booleans have been left out deliberately or left out accidentally. That's probably best, I'll make two new
    //  functions for each of those when I have some time.

    @Override
    public List<String> getOptionsList() {
        List<String> list = new ArrayList<>();
        for(Map.Entry<String, List<Object>> entry : paramMap.entrySet()) {
            String name = entry.getKey();
            List<Object> paramValues = entry.getValue();
            for(Object paramValue : paramValues) {
                list.add(StrUtils.flagify(name));
                list.add(StrUtils.toOptionValue(paramValue));
            }
        }
        return list;
    }

    /**
     * Goes through a list of strings and builds the corresponding parameter set
     * @param options
     * @throws Exception
     */
    @Override
    public void setOptionsList(final List<String> options) throws
        Exception {
        for(int i = 0; i < options.size(); i++) {
            String option = options.get(i);
            String flag = StrUtils.unflagify(option);
            // if the flag is an option (i.e. key value pair, not just a flag)
            if(StrUtils.isOption(option, options)) {
                // for example, "-d "DTW -w 5""
                // get the next value as the option value and split it into sub options
                // the example would be split into ["DTW", "-w", "5"]
                String[] subOptions = Utils.splitOptions(options.get(++i));
                options.set(i, "");
                options.set(i - 1, "");
                // the 0th element is the main option value
                // in the example this is "DTW"
                String optionValue = subOptions[0];
                subOptions[0] = "";
                // get the value from str form
                Object value = StrUtils.fromOptionValue(optionValue);
                if(value instanceof ParamHandler) {
                    // handle sub parameters
                    ParamSet paramSet = new ParamSet();
                    // subOptions contains only the parameters for the option value
                    // in the example this is ["-w", "5"]
                    // set these suboptions for the parameter value
                    paramSet.setOptions(subOptions);
                    // add the parameter to this paramSet with correspond value (example "DTW") and corresponding
                    // sub options / parameters for that value (example "-w 5")
                    add(flag, value, paramSet);
                } else {
                    // the parameter is raw, i.e. "-a 6" <-- 6 has no parameters, therefore is raw
                    add(flag, value);
                }
            } else {
                // assume all flags are represented using boolean values
                add(flag, true);
            }
            options.set(i, "");
        }
    }

    @Override
    public ParamSet getParams() {
        return this;
    }

    @Override
    public void setParams(final ParamSet param) {
        for(Entry<String, List<Object>> entry : param.paramMap.entrySet()) {
            String key = entry.getKey();
            List<Object> value = entry.getValue();
            add(key, value);
        }
    }

    @Override
    public String toString() {
        return StrUtils.join(", ", getOptions());
    }

}
