.onLoad <- function(libname, pkgname) {
  # Ensure TensorFlow is installed
  if (!reticulate::py_module_available("tensorflow")) {
    reticulate::py_install("tensorflow")
  }

  Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")  # optional: disable GPU

  # Import TensorFlow once when package loads
  assign("tf", reticulate::import("tensorflow"), envir = parent.env(environment()))
}
