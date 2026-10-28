/** Navigation between the game, the explorer, the game review and the about page (implemented by App). */
export interface Nav {
    current: 'play' | 'explore' | 'review' | 'about';
    play(): void;
    explorer(): void;
    /** The review page with the last finished game. */
    review(): void;
    about(): void;
}
