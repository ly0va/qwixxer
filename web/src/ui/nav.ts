/** Navigation between the game, the explorer and the game review (implemented by App). */
export interface Nav {
    current: 'play' | 'explore' | 'review';
    play(): void;
    explorer(): void;
    /** The review page with the last finished game. */
    review(): void;
}
