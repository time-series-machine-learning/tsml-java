package utilities;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class FileUtils {


    public static void writeToFile(String str, String path) throws
                                                            IOException {
        makeParentDir(path);
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        writer.write(str);
        writer.close();
    }

    public static void writeToFile(String str, File file) throws
                                                          IOException {
        writeToFile(str, file.getPath());
    }

    public static String readFromFile(String path) throws
                                                   IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        String line;
        StringBuilder builder = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            builder.append(line);
            builder.append(System.lineSeparator());
        }
        reader.close();
        return builder.toString();
    }

    public static String readFromFile(File path) throws
                                                 IOException {
        return readFromFile(path.getPath());
    }

    public static void makeParentDir(String filePath) {
        makeParentDir(new File(filePath));
    }

    public static void makeParentDir(File file) {
        File parent = file.getParentFile();
        if(parent != null) {
            parent.mkdirs();
        }
    }

    public static class FileLocker {
        private Thread thread;
        private final File file;
        private static final long interval = TimeUnit.MILLISECONDS.convert(1, TimeUnit.MINUTES);
        private static final long expiredInterval = interval + TimeUnit.MILLISECONDS.convert(1, TimeUnit.MINUTES);
        private final AtomicBoolean unlocked = new AtomicBoolean(false);

        public File getFile() {
            return file;
        }

        public static FileLocker lock(String path) throws
                                                   IOException {
            return lock(new File(path));
        }

        public static FileLocker lock(File file) throws
                                                 IOException {
            file = new File(file.getPath() + ".lock");
            makeParentDir(file);
            boolean stop = false;
            while(!stop) {
                boolean created = file.createNewFile();
                if(created) {
//                    System.out.println(file.getPath() + " created");
                    file.deleteOnExit();
                    return new FileLocker(file);
                } else {
                    long lastModified = file.lastModified();
                    if(lastModified <= 0) {
//                        System.out.println(file.getPath() + " lm < 0");
                        stop = true;
                    } else if(lastModified + expiredInterval < System.currentTimeMillis()) {
//                        System.out.println(file.getPath() + " lm > interval" + lastModified);
                        stop = !file.delete();
                    } else {
//                        System.out.println(file.getPath() + " lm < interval " + lastModified + " " + file.exists());
                        stop = true;
                    }
                }
            }
            return null;
        }

        private void watch() {
            do {
                boolean setLastModified = file.setLastModified(System.currentTimeMillis());
                if(!setLastModified) {
                    unlocked.set(true);
                }
                if(!unlocked.get()) {
                    try {
                        Thread.sleep(interval);
                    } catch (InterruptedException e) {
                        unlocked.set(true);
                    }
                }
            } while (!unlocked.get());
            file.delete();
        }

        private FileLocker(final File file, boolean failed) {
            this.file = file;
            if(failed) {
                // the failed to lock version, i.e. the locker is already unlocked.
                unlocked.set(true);
            } else {
                setup();
            }

        }

        private FileLocker(final File file) {
            this(file, false);
        }

        private void setup() {
            thread = new Thread(this::watch);
            thread.setDaemon(true);
            thread.start();
        }

        public void unlock() {
            unlocked.set(true);
            if(thread != null) thread.interrupt();
        }

        public boolean isUnlocked() {
            return unlocked.get();
        }

        public boolean isLocked() {
            return !isUnlocked();
        }
    }
}
