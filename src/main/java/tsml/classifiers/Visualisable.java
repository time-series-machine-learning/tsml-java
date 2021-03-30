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

package tsml.classifiers;

import java.io.File;

/**
 * Interface for classifiers that can output visualisations of the final model.
 *
 * @author Matthew Middlehurst
 **/
public interface Visualisable {

    /**
     * Stores a path to save visualisation files to.
     *
     * @param path String directory path
     * @return true if path is valid, false otherwise.
     */
    boolean setVisualisationSavePath(String path);

    /**
     * Create model visualisations and save them to a set path.
     *
     * @return true if successful, false otherwise
     * @throws Exception if failure to set path or create visualisation
     */
    boolean createVisualisation() throws Exception;

    /**
     * Create a directory at a given path.
     *
     * @param path String directory path
     * @return true if folder is created successfully, false otherwise
     */
    default boolean createVisualisationDirectories(String path) {
        File f = new File(path);
        boolean success = true;
        if (!f.isDirectory())
            success = f.mkdirs();
        return success;
    }
}