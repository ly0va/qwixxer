/** Navigation between the game and the explorer (implemented by App). */
export interface Nav {
    current: 'play' | 'explore';
    play(): void;
    explorer(): void;
}
