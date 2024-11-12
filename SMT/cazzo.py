import threading
import time
import curses

# Funzione principale che stampa progressivamente sul terminale
def main_function(stdscr):
    for i in range(10):  # Simuliamo una funzione lunga
        stdscr.addstr(i, 0, f"Processo principale in esecuzione: step {i+1}\n")
        stdscr.refresh()
        time.sleep(2)  # Simula un'operazione che richiede tempo

# Funzione per il timer che mostra il tempo rimanente in basso
def countdown_timer(stdscr, duration):
    max_y, max_x = stdscr.getmaxyx()  # Ottieni le dimensioni del terminale
    timer_y = max_y - 1  # Posiziona il timer sulla riga in fondo

    for remaining in range(duration, 0, -1):
        stdscr.addstr(timer_y, 0, f"Tempo rimanente: {remaining} secondi     ")
        stdscr.refresh()
        time.sleep(1)

    stdscr.addstr(timer_y, 0, "Tempo rimanente: Completato!               ")
    stdscr.refresh()

# Funzione wrapper per gestire curses e i thread
def run_with_timer(stdscr):
    timer_duration = 20  # in secondi

    # Creiamo e avviamo i thread per la funzione principale e il timer
    main_thread = threading.Thread(target=main_function, args=(stdscr,))
    timer_thread = threading.Thread(target=countdown_timer, args=(stdscr, timer_duration))

    main_thread.start()
    timer_thread.start()

    # Attendiamo che i thread finiscano
    main_thread.join()
    timer_thread.join()

    # Mostra messaggio di fine
    max_y, _ = stdscr.getmaxyx()
    stdscr.addstr(max_y - 2, 0, "Esecuzione completa! Premi un tasto per uscire.")
    stdscr.refresh()
    stdscr.getch()

# Avvia curses e il programma
curses.wrapper(run_with_timer)
