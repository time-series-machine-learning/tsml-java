/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.collections.params;

import weka.core.OptionHandler;

import java.util.*;

/**
 * Purpose: handle generic options for a class. These may be in the form of a String[] (weka style), List<String> or
 * bespoke ParamSet / ParamSpace themselves.
 *
 * You must override the getParams() and setParams() functions. These will be automatically transformed into list /
 * array format for compatibility with non-bespoke non-ParamSet / ParamSpace code.
 *
 * Contributors: goastler
 */
public interface ParamHandler
    extends OptionHandler {
    
    /**
     * get the options array
     * @return
     */
    @Override
    default String[] getOptions() {
        return getOptionsList().toArray(new String[0]);
    }

    /**
     * get the options list.
     * @return
     */
    default List<String> getOptionsList() {
        return getParams().getOptionsList();
    }

    /**
     * set options via list.
     * @param options
     * @throws Exception
     */
    default void setOptions(List<String> options) throws
                                                      Exception {
        ParamSet params = new ParamSet();
        params.setOptions(options);
        setParams(params);
    }

    /**
     * set options via array.
     * @param options the list of options as an array of strings
     * @throws Exception
     */
    @Override
    default void setOptions(String[] options) throws
                                      Exception {
        setOptions(Arrays.asList(options));
    }

    @Override
    default Enumeration listOptions() {
        return Collections.enumeration(listParams());
    }

    default void setParams(ParamSet paramSet) throws Exception {
        // OVERRIDE THIS
    }

    default ParamSet getParams() {
        // OVERRIDE THIS
        return new ParamSet();
    }

    default Set<String> listParams() {
        // todo use getParams to populate this
        throw new UnsupportedOperationException("param list not specified");
    }

}
