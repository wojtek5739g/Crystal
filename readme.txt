Projekt został wykonany przy pomocy Pythona w wersji 3.11.6

Za pomocą tej komendy w powershellu została wykonana instrukcja uruchamiająca symulacje po kolei dla wszystkich plików z parametrami:
python crystal.py 'parameters5.txt' ; python crystal.py 'parameters6.txt' ; python crystal.py 'parameters7.txt' ; python crystal.py 'parameters8.txt' ; python crystal.py 'parameters9.txt' ; python crystal.py 'parameters10.txt'

Można też użyć zwięźlejszej formy: 1..6 | ForEach-Object { python crystal.py "parameters$_.txt" }

Następnie po wykonaniu symulacji (w tym zebranie czasów poszczególnych symulacji) python plot_times.py