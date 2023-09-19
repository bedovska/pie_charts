
import re
import asyncio

import pandas as pd

from datetime import datetime
from typing import (List, Optional)
from pathlib import Path
from fire import Fire
from tqdm.cli import tqdm
from contextlib import suppress
from telethon import TelegramClient
from telethon import functions, types
from telethon.types import ChannelForbidden
from telethon.errors import SessionPasswordNeededError, PhoneNumberInvalidError
from telethon.errors.rpcerrorlist import ChannelPrivateError, ChannelInvalidError, InviteHashExpiredError
# from telethon.tl.functions.messages.check_chat_invite import CheckChatInviteRequest

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




async def check_channel_exists(channel_username):
    try:
        channel = await client.get_entity(channel_username)
        return True
    except ChannelPrivateError:
        return False
    except ChannelInvalidError:
        return False
    except InviteHashExpiredError:
        return False
    except ValueError:
        return False



async def download_messages(
    # client: TelegramClient,
    channel_name: str,
    date_from: datetime,
    keyword: str,
) -> Optional[pd.DataFrame]:

    messages = pd.DataFrame()

    channel_exists = await check_channel_exists(channel_name)

    if not channel_exists:
        print(f"Channel does not exists: {channel_name}")
        return None

    # try:
    # async with client.takeout() as takeout:
    #    async for msg in takeout.iter_messages(
    
    async for msg in client.iter_messages(
            channel_name,
            search=keyword,
            reverse=True,
            # offset_date=date_from,
            wait_time=0,
        ):

            if msg.reactions is not None:
                reactions = {r.reaction.emoticon: r.count for r in msg.reactions.results}
            else:
                reactions = {}

            if msg.forward is not None:
                not_hidden = msg.forward.chat is not None
                not_forbidden = not_hidden and not isinstance(msg.forward.chat, ChannelForbidden)

                is_forwarded = True
                forward_from = f"https://t.me/{msg.forward.chat.username}" \
                               if not_hidden and not_forbidden else ""
            else:
                is_forwarded = False
                forward_from = ""

            msg_item = {
                "id": msg.id,
                "date": msg.date.isoformat(),
                "views": msg.views,
                "reactions": reactions,
                "has_photo": msg.photo is not None,
                "has_video": msg.video is not None,
                "text": msg.text,
                "cur_date": datetime.now().isoformat(),
                "is_forwarded": is_forwarded,
                "forward_from": forward_from,
            }
            
            messages = messages.append(msg_item, ignore_index=True)
    

    if len(messages) == 0:
        return None 

    messages =  messages.sort_values(by="date", ascending=False)


    # except ValueError:
    #     print(f"Error during message download: {channel_name}")
    #     return None
    # except Exception:
    #     print(f"Error during message download: {channel_name}")
    #     return None

    return messages


async def download_channels(
    done: List[str],
    waitlist: List[str],
    out_folder: Path,
    date_from: datetime,
    search_keyword: str,
) -> None:
    await client.start()    
    await client.connect()

    while True:
        if len(waitlist) == 0:
            break

        channel_name = waitlist.pop(0)
        done.append(channel_name)

        
        print(f"Current channel: {channel_name}")
        print(f"Done: {len(done)}, waitlist: {len(waitlist)}")

        messages = await download_messages(
            channel_name,
            date_from,
            search_keyword
        )

        if messages is None:
            continue

        print(f"Num of messages in channel with keyword {search_keyword}: " \
              f"{len(messages)}")

        print()

        channel_short_name = channel_name.split("/")[-1]
        messages.to_csv(out_folder / f"{channel_short_name}.csv", index=False)



def load_done(folder: Path) -> List[str]:
    short_names = [fn.stem for fn in folder.glob("*.csv")]
    done = [f"https://t.me/{sn}" for sn in short_names]
    return done


def load_waitlist(folder: Path) -> List[str]:
    df = pd.read_csv("df.csv")
    waitlist = df['link'].tolist()
    return waitlist


api_id = 
api_hash = ""

client = TelegramClient("o", api_id, api_hash)


def main():
    date_from = datetime(year=2023, month=5, day=1)
    search_keyword = "выборы"

    download_folder = Path("download")
    done_channels = load_done(download_folder) 
    waitlist_channels = load_waitlist(download_folder) 
    waitlist_channels = [ch for ch in waitlist_channels if ch not in done_channels]

    print(f"Num of channels done: {len(done_channels)}")
    print(f"Num of channels in waitlist: {len(waitlist_channels)}")
    print()


    client.loop.run_until_complete(download_channels(
        done_channels,
        waitlist_channels,
        download_folder,
        date_from,
        search_keyword,
    ))


if __name__ == "__main__":
    Fire(main)
