/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package utilities;

import utilities.StringUtilities;
import weka.core.Utils;

import java.util.*;

public class ParamSet
    implements ParamHandler {
    private Object value;
    private Map<String, List<ParamSet>> map = new LinkedHashMap<>();

    public ParamSet() {}

    public ParamSet(String[] options) throws Exception {
        setOptions(options);
    }

    public ParamSet(String str) throws
                                    Exception {
        this(Utils.splitOptions(str));
    }

    public Object getValue() {
        return value;
    }

    public void setValue(Object value) {
        this.value = value;
    }

    public void add(String name, ParamSet sub) {
        map.computeIfAbsent(name, k -> new ArrayList<>()).add(sub);
    }

    public void add(String name, Object value) {
        ParamSet paramSet = new ParamSet();
        paramSet.setValue(value);
        add(name, paramSet);
    }

    public void add(String name) {
        add(name, null);
    }

    public void clear() {
        map.clear();
        value = null;
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setOptionsList(List<String> options) throws Exception {
        int i = 0;
        if(!StringUtilities.isFlag(options.get(i))) {
            i++;
            setValue(options.get(0));
        }
        for(; i < options.size(); i++) {
            if(StringUtilities.isFlag(options.get(i))) {
                if(i + 1 >= options.size() || StringUtilities.isFlag(options.get(i + 1))) {
                    add(options.get(i));
                } else {
                    ParamSet permutation = new ParamSet();
                    permutation.setOptions(Utils.splitOptions(options.get(i + 1)));
                    add(options.get(i), permutation);
                    i++;
                }
            }
        }
    }

    @Override
    public List<String> getOptionsList() {
        List<String> options = new ArrayList<>();
        if(value != null) {
            String valueStr = StringUtilities.toOptionValue(value);
            options.add(valueStr);
        }
        for(Map.Entry<String, List<ParamSet>> entry : map.entrySet()) {
            String name = entry.getKey();
            List<ParamSet> subs = entry.getValue();
            for(ParamSet sub : subs) {
                if(sub != null) StringUtilities.addOption(name, options, sub.toString());
            }
        }
        return options;
    }

    @Override
    public String toString() {
        String[] options = getOptions();
        if(options.length > 1) {
            return Utils.joinOptions(options);
        } else {
            return options[0];
        }
    }

    public static void main(String[] args) throws
                                           Exception {
        ParamSet parameterSet = new ParamSet();
        ParamSet parameterSet1 = new ParamSet("-a \"d -b c\"");
        System.out.println(parameterSet1);
        parameterSet.add("-K", parameterSet1);
        ParamSet par = new ParamSet();
        par.add("-M", parameterSet);
        System.out.println(parameterSet);
        System.out.println(par);
//        ParamSet p = new ParamSet();
//        p.setOptions(Utils.splitOptions("-x \"y -e f -g h\" -z b -c d "));
//
//        System.out.println(p);
//
//        ParamSet parent = new ParamSet();
//        parent.add("-z", "b");
//        parent.add("-a");
//        parent.add("-c", "d");
//        ParamSet child = new ParamSet("y");
//        child.add("-e", "f");
//        child.add("-g", "h");
//        parent.add("-x", child);
//        String[] options = parent.getOptions();
//        String optionsStr = Utils.joinOptions(options);
//        System.out.println(optionsStr);
//        parent.clear();
//        parent.setOptions(Utils.splitOptions(optionsStr));
//        options = parent.getOptions();
//        optionsStr = Utils.joinOptions(options);
//        System.out.println(optionsStr);
    }
}
