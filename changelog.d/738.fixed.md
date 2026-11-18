`ArchimedeanSpiral.render` no longer raises "scan body function carry input and carry output must have equal types" on a float32 grid with float64 parameters: the scan carry takes the promoted dtype.
