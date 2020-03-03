package utilities.params;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import utilities.StrUtils;
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
public class ParamSet implements ParamHandler {

    private Map<String, List<Object>> paramMap = new HashMap<>();

    public static void main(String[] args) {
        ParamSet oParamSet = new ParamSet();
        oParamSet.add(DTW.getWarpingWindowFlag(), 3);
        ParamSet paramSet = new ParamSet();
        paramSet.add(DistanceMeasureable.getDistanceFunctionFlag(), new DTWDistance(), oParamSet);
        String[] options;
        options = oParamSet.getOptions();
        System.out.println(Utils.joinOptions(options));
        options = paramSet.getOptions();
        System.out.println(Utils.joinOptions(options));
    }

    public List<Object> get(String name) {
        return paramMap.get(name);
    }

    public ParamSet add(String name, Object value) {
        paramMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
        return this;
    }

    public ParamSet add(String name, ParamHandler value, List<ParamSet> params) {
        for(ParamSet paramSet : params) {
            add(name, value, paramSet);
        }
        return this;
    }

    /**
     * add a parameter name mapping to a parameter value. The value is a parameter handler which accepts the ParamSet
     * . The ParamSet is applied to the value during this function. An example would be ("DTW", new DTW(), dtwParams)
     * . The DTW instance accepts the dtwParams object which sets, say, the warping window. Back here, the "DTW"
     * string is mapped to the DTW instance. All parameters / configuration are therefore housed in a map using
     * containment.
     * @param name
     * @param value
     * @param param
     * @return
     */
    public ParamSet add(String name, ParamHandler value, ParamSet param) {
        value.setParams(param);
        return add(name, value);
    }

    public ParamSet addAll(ParamSet paramSet) {
        paramSet.paramMap.forEach(this::add);
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
        paramMap.forEach((name, paramValues) -> {
            paramValues.forEach(paramValue -> {
                list.add(StrUtils.flagify(name));
                list.add(StrUtils.toOptionValue(paramValue));
            });
        });
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
            // if the flag is an option (i.e. key value pair, not just a flag, see above comment)
            if(StrUtils.isOption(option, options)) {
                String[] subOptions = Utils.splitOptions(option);
                String optionValue = subOptions[0];
                subOptions[0] = "";
                // get the value
                Object value = StrUtils.fromOptionValue(optionValue);
                if(value instanceof ParamHandler) {
                    // handle sub parameters
                    ParamSet paramSet = new ParamSet();
                    paramSet.setOptions(subOptions);
                    add(flag, (ParamHandler) value, paramSet);
                } else {
                    add(flag, value);
                }
                options.set(i, "");
                i++;
            } else {
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
        param.paramMap.forEach(this::add);
    }

    @Override
    public String toString() {
        return "ParamSet" +
            "{" +
            //            "paramMap=" +
            //                " \"" +
            StrUtils.join(", ", getOptions()) +
            //                " \""
            +'}'
            ;
    }

    // todo handle lists as the value of param value --> i.e. split list into separate param values
}
