import init, { WasmExplorer, WasmGame } from '../../pkg/qwixxer_web.js';
import wasmUrl from '../../pkg/qwixxer_web_bg.wasm?url';
import { serve, type Request } from './serve';

const handle = serve({ init: () => init({ module_or_path: wasmUrl }), WasmGame, WasmExplorer }, (r) => self.postMessage(r));
self.onmessage = (e: MessageEvent<Request>) => handle(e.data);
